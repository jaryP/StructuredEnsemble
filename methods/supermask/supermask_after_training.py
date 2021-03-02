import logging
import os
from collections import defaultdict
from copy import deepcopy

import dill as pickle
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from eval import eval_model
from methods.base import EnsembleMethod
from methods.supermask.base import mask_training, iterative_mask_training, \
    be_model_training
from methods.supermask.models_utils import extract_inner_model, \
    remove_wrappers_from_model, add_wrappers_to_model, \
    get_masks_from_gradients

from methods.supermask.layers import EnsembleMaskedWrapper, \
    BatchEnsembleMaskedWrapper
from utils import train_model, calculate_trainable_parameters


class BatchPruningSuperMaskPostTraining(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.ensemble = method_parameters['n_ensemble']
        self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']
        self.divergence_w = method_parameters.get('divergence_w', 1)

        self.max_prune_percentage = method_parameters['max_prune_percentage']
        self.global_pruning = method_parameters['global_pruning']
        self.score_tolerance = method_parameters['score_tolerance']
        # self.re_init = method_parametersscore_tolerance
        self.grad_reduce = method_parameters.get('grad_reduce', 'mean')
        self.lr = method_parameters.get('lr', 0.001)

        # self.shrink_iterations = method_parameters.get('shrink_iterations', 0)
        # self.shrink_pruning = method_parameters.get('shrink_pruning',
        #                                             self.prune_percentage)

        self.models = torch.nn.ModuleList()
        self.model = deepcopy(model)
        self.distributions = torch.nn.ModuleDict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset,
                     optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        assert eval_dataset is not None

        logger = logging.getLogger(__name__)

        optim = optimizer(
            [param for name, param in self.model.named_parameters() if
             param.requires_grad])
        train_scheduler = scheduler(optim)

        logger.info('Training base model')
        best_model, scores, best_model_scores, losses = train_model(
            model=self.model,
            optimizer=optim,
            epochs=epochs,
            train_loader=train_dataset,
            scheduler=train_scheduler,
            early_stopping=early_stopping,
            test_loader=test_dataset,
            eval_loader=eval_dataset,
            device=self.device)
        self.model.load_state_dict(best_model)

        base_eval_scores, _ = eval_model(self.model, eval_dataset, topk=[1, 5],
                                         device=self.device)
        base_eval_scores = base_eval_scores[1]

        add_wrappers_to_model(self.model,
                              masks_params=self.supermask_parameters,
                              ensemble=self.ensemble, batch_ensemble=True)

        params = [param for name, param in self.model.named_parameters()
                  if param.requires_grad and 'distributions' in name]

        for _, module in self.model.named_modules():
            if isinstance(module, _BatchNorm):
                params.extend(module.named_parameters())

        optim = Adam([param for name, param in self.model.named_parameters()
                      if param.requires_grad and 'distributions' in name],
                     self.lr)

        train_scheduler = scheduler(optim)

        logger.info('Training masks model')

        best_model, scores, best_model_scores, losses = be_model_training(
            model=self.model, optimizer=optim,
            epochs=self.mask_epochs,
            train_loader=train_dataset,
            scheduler=train_scheduler,
            early_stopping=early_stopping,
            test_loader=test_dataset,
            eval_loader=eval_dataset,
            device=self.device, w=self.divergence_w)

        self.model.load_state_dict(best_model)

        grads = defaultdict(lambda: defaultdict(list))

        for i, (x, y) in enumerate(train_dataset):
            x, y = x.to(self.device), y.to(self.device)
            bs = x.shape[0]
            x = torch.cat([x for _ in range(self.ensemble)], dim=0)

            outputs = self.model(x)
            outputs = outputs.view([self.ensemble, bs, -1])
            pred = torch.mean(outputs, 0)

            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

            self.model.zero_grad()
            loss.backward(retain_graph=True)

            for name, module in self.model.named_modules():
                if isinstance(module, BatchEnsembleMaskedWrapper):
                    grad = torch.autograd.grad(loss, module.last_mask,
                                               retain_graph=True)
                    for i, g in enumerate(grad):
                        grads[i][name].append(torch.abs(g).cpu())

        self.model.zero_grad()

        remove_wrappers_from_model(self.model)

        i = 0
        for mi, ens_grads in tqdm(grads.items(),
                                  desc='Extracting inner models'):

            logger.info('Extracting model {}'.format(i))
            i += 1

            f = lambda x: torch.mean(x, 0)

            if self.grad_reduce == 'median':
                f = lambda x: torch.median(x, 0)
            elif self.grad_reduce == 'max':
                f = lambda x: torch.max(x, 0)

            ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs
                         in ens_grads.items()}

            best_model = None
            prunings = []

            max_pruning = min(self.max_prune_percentage, 1)
            min_pruning = 0

            while True:
                local_pruning = (max_pruning + min_pruning) / 2

                logger.info('Pruning percentages: min:{}, max:{}, p:{}'.format(
                    min_pruning, max_pruning, local_pruning))

                if abs(max_pruning - min_pruning) <= 0.01:
                    break

                prunings.append(local_pruning)

                masks = get_masks_from_gradients(gradients=ens_grads,
                                                 prune_percentage=local_pruning,
                                                 global_pruning=self.global_pruning,
                                                 device=self.device)

                local_model = extract_inner_model(self.model, masks,
                                                  re_init=False)

                eval_scores, _ = eval_model(local_model, eval_dataset,
                                            topk=[1, 5],
                                            device=self.device)

                if best_model is None:
                    best_model = local_model

                logger.info('Scores: best {}, current: {} {} {}'.format(
                    base_eval_scores, eval_scores[1], base_eval_scores - eval_scores[1], base_eval_scores * self.score_tolerance))

                if base_eval_scores - eval_scores[1] <= \
                        base_eval_scores * self.score_tolerance:
                    best_model = local_model
                    min_pruning = local_pruning
                else:
                    max_pruning = local_pruning

            self.models.append(best_model.to('cpu'))

        #
        # if self.shrink_iterations == 0:
        #     bar = tqdm(range(self.shrink_iterations + 1),
        #                desc='Shrink Iterations', disable=True)
        # else:
        #     bar = tqdm(range(self.shrink_iterations + 1),
        #                desc='Shrink Iterations')
        #
        # for i in bar:
        #     last_iteration = i == self.shrink_iterations
        #
        #     if last_iteration:
        #         pruning = self.prune_percentage
        #         ens = self.ensemble
        #     else:
        #         pruning = self.shrink_pruning
        #         ens = 1
        #
        #     add_wrappers_to_model(self.model,
        #                           masks_params=self.supermask_parameters,
        #                           ensemble=ens, batch_ensemble=True)
        #
        #     params = [param for name, param in self.model.named_parameters()
        #               if param.requires_grad and 'distributions' in name]
        #
        #     for _, module in self.model.named_modules():
        #         if isinstance(module, _BatchNorm):
        #             params.extend(module.named_parameters())
        #
        #     optim = Adam([param for name, param in self.model.named_parameters()
        #                   if param.requires_grad and 'distributions' in name],
        #                  self.lr)
        #
        #     train_scheduler = scheduler(optim)
        #
        #     best_model, scores, best_model_scores, losses = be_model_training(
        #         model=self.model, optimizer=optim,
        #         epochs=self.mask_epochs,
        #         train_loader=train_dataset,
        #         scheduler=train_scheduler,
        #         early_stopping=early_stopping,
        #         test_loader=test_dataset,
        #         eval_loader=eval_dataset,
        #         device=self.device, w=self.divergence_w)
        #     self.model.load_state_dict(best_model)
        #
        #     for _, module in self.model.named_modules():
        #         if isinstance(module, _BatchNorm):
        #             module.reset_parameters()
        #
        #     # print(self.model)
        #
        #     grads = defaultdict(lambda: defaultdict(list))
        #
        #     for i, (x, y) in enumerate(train_dataset):
        #         x, y = x.to(self.device), y.to(self.device)
        #         bs = x.shape[0]
        #         x = torch.cat([x for _ in range(ens)], dim=0)
        #
        #         outputs = self.model(x)
        #         outputs = outputs.view([ens, bs, -1])
        #         pred = torch.mean(outputs, 0)
        #
        #         loss = torch.nn.functional.cross_entropy(pred, y,
        #                                                  reduction='mean')
        #
        #         self.model.zero_grad()
        #         loss.backward(retain_graph=True)
        #
        #         for name, module in self.model.named_modules():
        #             if isinstance(module, BatchEnsembleMaskedWrapper):
        #                 grad = torch.autograd.grad(loss, module.last_mask,
        #                                            retain_graph=True)
        #                 if last_iteration:
        #                     for i, g in enumerate(grad):
        #                         grads[i][name].append(torch.abs(g).cpu())
        #                 else:
        #                     grads[0][name].append(torch.abs(
        #                         torch.mean(torch.stack(grad, 0), 0)).cpu())
        #
        #     self.model.zero_grad()
        #
        #     remove_wrappers_from_model(self.model)
        #
        #     for mi, ens_grads in tqdm(grads.items(),
        #                               desc='Extracting inner models'):
        #         f = lambda x: torch.mean(x, 0)
        #
        #         if self.grad_reduce == 'median':
        #             f = lambda x: torch.median(x, 0)
        #         elif self.grad_reduce == 'max':
        #             f = lambda x: torch.max(x, 0)
        #
        #         ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for
        #                      name, gs in ens_grads.items()}
        #
        #         # classifier for vgg
        #         # fc for ResNet
        #         # print(ens_grads['fc'])
        #
        #         masks = get_masks_from_gradients(gradients=ens_grads,
        #                                          prune_percentage=pruning,
        #                                          global_pruning=self.global_pruning,
        #                                          device=self.device)
        #
        #         model = extract_inner_model(self.model, masks,
        #                                     re_init=self.re_init)
        #
        #         if last_iteration:
        #             self.models.append(model)
        #         else:
        #             self.model = model
        #
        #         p = calculate_trainable_parameters(model)
        #         print(mi, last_iteration, p,
        #               calculate_trainable_parameters(self.model))
        #
        # all_scores = []
        #
        # for i in tqdm(range(len(self.models)), desc='Training models'):
        #     model = self.models[i]
        #
        #     # print(model)
        #
        #     optim = optimizer(
        #         [param for name, param in model.named_parameters() if
        #          param.requires_grad])
        #
        #     train_scheduler = scheduler(optim)
        #
        #     best_model, scores, best_model_scores, losses = train_model(
        #         model=model, optimizer=optim,
        #         epochs=epochs, train_loader=train_dataset,
        #         scheduler=train_scheduler,
        #         early_stopping=early_stopping,
        #         test_loader=test_dataset,
        #         eval_loader=eval_dataset,
        #         device=self.device)
        #
        #     model.load_state_dict(best_model)
        #     model.to('cpu')
        #     all_scores.append(scores)

        self.device = 'cpu'
        return None

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        outputs = torch.stack([m(x) for m in self.models], 1)
        if reduce:
            outputs = torch.mean(outputs, 1)
        return outputs

    def load(self, path):
        self.device = 'cpu'
        for i in range(self.ensemble):
            with open(os.path.join(path, 'model_{}.pt'.format(i)),
                      'rb') as file:
                m = pickle.load(file)
            # m = torch.load(os.path.join(path, 'model_{}.pt'.format(i)), map_location=self.device)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        for i, m in enumerate(self.models):
            with open(os.path.join(path, 'model_{}.pt'.format(i)),
                      'wb') as file:
                pickle.dump(m, file)
            # torch.save(m, os.path.join(path, 'model_{}.pt'.format(i)))
