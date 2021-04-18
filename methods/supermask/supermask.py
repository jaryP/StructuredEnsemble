import os
from collections import defaultdict
from copy import deepcopy

import dill as pickle
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam
from tqdm import tqdm

from methods.base import EnsembleMethod
from methods.supermask.base import be_model_training
from methods.supermask.models_utils import extract_inner_model, \
    remove_wrappers_from_model, add_wrappers_to_model, \
    get_masks_from_gradients

from methods.supermask.layers import BatchEnsembleMaskedWrapper
from utils import train_model


class ExtremeBatchPruningSuperMask(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.ensemble = method_parameters['n_ensemble']
        self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']
        self.divergence_w = method_parameters.get('divergence_w', 1)

        self.prune_percentage = method_parameters['prune_percentage']
        self.re_init = method_parameters['re_init']
        self.global_pruning = method_parameters['global_pruning']
        self.grad_reduce = method_parameters.get('grad_reduce', 'mean')
        self.lr = method_parameters.get('lr', 0.001)

        self.shrink_iterations = method_parameters.get('shrink_iterations', 0)
        self.shrink_pruning = method_parameters.get('shrink_pruning', self.prune_percentage)

        self.models = torch.nn.ModuleList()
        self.model = deepcopy(model)
        self.distributions = torch.nn.ModuleDict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        if self.shrink_iterations == 0:
            bar = tqdm(range(self.shrink_iterations + 1), desc='Shrink Iterations', disable=True)
        else:
            bar = tqdm(range(self.shrink_iterations + 1), desc='Shrink Iterations')

        for i in bar:
            last_iteration = i == self.shrink_iterations

            if last_iteration:
                pruning = self.prune_percentage
                ens = self.ensemble
            else:
                pruning = self.shrink_pruning
                ens = 1

            add_wrappers_to_model(self.model, masks_params=self.supermask_parameters,
                                  ensemble=ens, batch_ensemble=True)

            params = [param for name, param in self.model.named_parameters()
                          if param.requires_grad and 'distributions' in name]

            for _, module in self.model.named_modules():
                if isinstance(module, _BatchNorm):
                    params.extend(module.named_parameters())

            optim = Adam([param for name, param in self.model.named_parameters()
                          if param.requires_grad and 'distributions' in name], self.lr)

            train_scheduler = scheduler(optim)

            best_model, scores, best_model_scores, losses = be_model_training(model=self.model, optimizer=optim,
                                                                              epochs=self.mask_epochs,
                                                                              train_loader=train_dataset,
                                                                              scheduler=train_scheduler,
                                                                              early_stopping=early_stopping,
                                                                              test_loader=test_dataset,
                                                                              eval_loader=eval_dataset,
                                                                              device=self.device, w=self.divergence_w)
            self.model.load_state_dict(best_model)

            for _, module in self.model.named_modules():
                if isinstance(module, _BatchNorm):
                    module.reset_parameters()

            grads = defaultdict(lambda: defaultdict(list))

            for i, (x, y) in enumerate(train_dataset):
                x, y = x.to(self.device), y.to(self.device)
                bs = x.shape[0]
                x = torch.cat([x for _ in range(ens)], dim=0)

                outputs = self.model(x)
                outputs = outputs.view([ens, bs, -1])
                pred = torch.mean(outputs, 0)

                loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                for name, module in self.model.named_modules():
                    if isinstance(module, BatchEnsembleMaskedWrapper):
                        grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                        if last_iteration:
                            for i, g in enumerate(grad):
                                grads[i][name].append(torch.abs(g).cpu())
                        else:
                            grads[0][name].append(torch.abs(torch.mean(torch.stack(grad, 0), 0)).cpu())

            self.model.zero_grad()

            remove_wrappers_from_model(self.model)

            for mi, ens_grads in tqdm(grads.items(), desc='Extracting inner models'):
                f = lambda x: torch.mean(x, 0)

                if self.grad_reduce == 'median':
                    f = lambda x: torch.median(x, 0)
                elif self.grad_reduce == 'max':
                    f = lambda x: torch.max(x, 0)

                ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in ens_grads.items()}

                masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=pruning,
                                                 global_pruning=self.global_pruning, device=self.device)

                model = extract_inner_model(self.model, masks, re_init=self.re_init)

                if last_iteration:
                    self.models.append(model)
                else:
                    self.model = model

                # p = calculate_trainable_parameters(model)
                # print(mi, last_iteration, p, calculate_trainable_parameters(self.model))

        all_scores = []

        for i in tqdm(range(len(self.models)), desc='Training models'):
            model = self.models[i]

            optim = optimizer([param for name, param in model.named_parameters() if param.requires_grad])

            train_scheduler = scheduler(optim)

            best_model, scores, best_model_scores, losses = train_model(model=model, optimizer=optim,
                                                                        epochs=epochs, train_loader=train_dataset,
                                                                        scheduler=train_scheduler,
                                                                        early_stopping=early_stopping,
                                                                        test_loader=test_dataset,
                                                                        eval_loader=eval_dataset,
                                                                        device=self.device)

            model.load_state_dict(best_model)
            all_scores.append(scores)
        return all_scores

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        outputs = torch.stack([m(x) for m in self.models], 1)
        if reduce:
            outputs = torch.mean(outputs, 1)
        return outputs

    def load(self, path):
        for i in range(self.ensemble):
            with open(os.path.join(path, 'model_{}.pt'.format(i)), 'rb') as file:
                m = pickle.load(file)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        for i, m in enumerate(self.models):
            with open(os.path.join(path, 'model_{}.pt'.format(i)), 'wb') as file:
                pickle.dump(m, file)