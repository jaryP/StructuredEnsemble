import os
from collections import defaultdict
from copy import deepcopy

import dill as pickle
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from methods.base import EnsembleMethod
from methods.supermask.base import mask_training, iterative_mask_training, be_model_training
from methods.supermask.models_utils import extract_inner_model, remove_wrappers_from_model, add_wrappers_to_model, \
    get_masks_from_gradients

from methods.supermask.layers import EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper
from utils import train_model, calculate_trainable_parameters


class GradSuperMask(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):

        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.ensemble = method_parameters['n_ensemble']

        self.prune_percentage = method_parameters['prune_percentage']
        self.re_init = method_parameters['re_init']
        self.global_pruning = method_parameters['global_pruning']

        self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']
        self.shrink_iterations = method_parameters.get('shrink_iterations', 0)
        self.shrink_pruning = method_parameters.get('shrink_pruning', self.prune_percentage)

        self.iterative_training = method_parameters['iterative_training']

        self.divergence = method_parameters.get('divergence', 'mmd')
        self.divergence_w = method_parameters.get('divergence_w', 1)

        self.grad_reduce = method_parameters.get('grad_reduce', 'mean')

        self.single_distribution = method_parameters.get('single_distribution', False)

        self.model = deepcopy(model)
        self.models = torch.nn.ModuleList()
        self.final_distributions = dict()
        self.initial_distributions = dict()

        if self.prune_percentage == 'adaptive':
            self.prune_percentage = 1 - (1 / self.ensemble)

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        add_wrappers_to_model(self.model, masks_params=self.supermask_parameters,
                              ensemble=self.ensemble, single_distribution=self.single_distribution)

        for name, module in self.model.named_modules():
            if isinstance(module, EnsembleMaskedWrapper):
                distr = module.distributions
                self.initial_distributions[name] = deepcopy(distr)

        if self.iterative_training and not self.single_distribution:
            iterative_mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
                                    dataset=train_dataset, ensemble=self.ensemble,
                                    divergence_type=self.divergence, w=self.divergence_w)
        else:
            mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
                          dataset=train_dataset, ensemble=self.ensemble, w=self.divergence_w)

        grads = defaultdict(lambda: defaultdict(list))

        for ens in range(self.ensemble):
            model = self.model

            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    module.set_distribution(ens)

            for i, (x, y) in enumerate(train_dataset):
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

                model.zero_grad()
                loss.backward(retain_graph=True)

                for name, module in model.named_modules():
                    if isinstance(module, EnsembleMaskedWrapper):
                        grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                        grad = grad[0]
                        grads[ens][name].append(torch.abs(grad).cpu())

            model.zero_grad()

        remove_wrappers_from_model(self.model)

        for ens in range(self.ensemble):
            ens_grads = grads[ens]

            f = lambda x: torch.mean(x, 0)

            if self.grad_reduce == 'median':
                f = lambda x: torch.median(x, 0)
            elif self.grad_reduce == 'max':
                f = lambda x: torch.max(x, 0)

            ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in ens_grads.items()}

            masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=self.prune_percentage,
                                             global_pruning=self.global_pruning, device=self.device)

            model = extract_inner_model(self.model, masks, re_init=self.re_init)

            self.models.append(model)

        for si in range(self.shrink_iterations):
            for mi, m in enumerate(self.models):
                add_wrappers_to_model(m, masks_params=self.supermask_parameters, ensemble=1)
                mask_training(epochs=self.mask_epochs, model=m, device=self.device,
                              dataset=train_dataset, ensemble=1)

                grads = defaultdict(list)
                for i, (x, y) in enumerate(train_dataset):
                    x, y = x.to(self.device), y.to(self.device)
                    pred = m(x)
                    loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

                    m.zero_grad()
                    loss.backward(retain_graph=True)

                    for name, module in m.named_modules():
                        if isinstance(module, EnsembleMaskedWrapper):
                            grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                            assert len(grad) == 1
                            grad = grad[0]
                            grads[name].append(torch.abs(grad).cpu())
                m.zero_grad()
                remove_wrappers_from_model(m)

                f = lambda x: torch.mean(x, 0)
                if self.grad_reduce == 'median':
                    f = lambda x: torch.median(x, 0)
                elif self.grad_reduce == 'max':
                    f = lambda x: torch.max(x, 0)

                ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in grads.items()}

                if self.global_pruning:
                    stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in ens_grads.items()])
                    grads_sum = np.sum(stacked_grads)
                    stacked_grads = stacked_grads / grads_sum

                    threshold = np.quantile(stacked_grads, q=self.prune_percentage)

                    masks = {name: torch.ge(gs / grads_sum, threshold).float().to(self.device)
                             for name, gs in ens_grads.items()}
                else:
                    masks = {name: torch.ge(gs, torch.quantile(gs, self.prune_percentage)).float()
                             for name, gs in ens_grads.items()}

                for name, mask in masks.items():
                    mask = mask.squeeze()
                    if mask.sum() == 0:
                        max = torch.argmax(ens_grads[name])
                        mask = torch.zeros_like(mask)
                        mask[max] = 1.0
                    masks[name] = mask

                remove_wrappers_from_model(m)
                p1 = calculate_trainable_parameters(m)
                m = extract_inner_model(m, masks, re_init=self.re_init)
                p2 = calculate_trainable_parameters(m)
                print(mi, p1, p2)

                self.models[mi] = m

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
        outputs = torch.stack([m(x) for m in self.models])
        if reduce:
            outputs = torch.mean(outputs, 0)
        return outputs

    def load(self, path):
        # self.final_distributions = torch.load(os.path.join(path, 'final_distributions.pt'), map_location=self.device)
        # self.initial_distributions = torch.load(os.path.join(path, 'initial_distributions.pt'),
        #                                         map_location=self.device)
        #
        # for k, v in self.final_distributions.items():
        #     plt.figure(k)
        #     ax = plt.subplot(121)
        #     d = v[0](size=5, reduce=False)
        #     d = d.squeeze().detach().cpu().numpy()
        #     sns.barplot(data=d, ax=ax)
        #
        #     ax = plt.subplot(122, sharey=ax)
        #     d = v[1](size=5, reduce=False)
        #     d = d.squeeze().detach().cpu().numpy()
        #     sns.barplot(data=d, ax=ax)

        # plt.show()

        for i in range(self.ensemble):
            m = torch.load(os.path.join(path, 'model_{}.pt'.format(i)), map_location=self.device)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        # torch.save(self.final_distributions, os.path.join(path, 'final_distributions.pt'))
        # torch.save(self.initial_distributions, os.path.join(path, 'initial_distributions.pt'))

        for i, m in enumerate(self.models):
            torch.save(m, os.path.join(path, 'model_{}.pt'.format(i)))


class BatchPruningSuperMask(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.ensemble = method_parameters['n_ensemble']
        self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']
        # self.train_before = method_parameters.get('train_before', True)
        self.divergence_w = method_parameters.get('divergence_w', 1)

        self.prune_percentage = method_parameters['prune_percentage']
        self.re_init = method_parameters['re_init']
        self.global_pruning = method_parameters['global_pruning']
        self.grad_reduce = method_parameters.get('grad_reduce', 'mean')

        self.models = torch.nn.ModuleList()
        self.model = deepcopy(model)
        self.distributions = torch.nn.ModuleDict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        add_wrappers_to_model(self.model, masks_params=self.supermask_parameters,
                              ensemble=self.ensemble, batch_ensemble=True)

        optim = Adam([param for name, param in self.model.named_parameters()
                      if param.requires_grad and 'distributions' in name], 0.001)

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

        grads = defaultdict(lambda: defaultdict(list))

        for i, (x, y) in enumerate(train_dataset):
            x, y = x.to(self.device), y.to(self.device)
            bs = x.shape[0]
            x = torch.cat([x for _ in range(self.ensemble)], dim=0)

            outputs = self.model(x)
            outputs = outputs.view([self.ensemble, bs, -1])
            pred = torch.mean(outputs, 0)

            loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

            self.model.zero_grad()
            loss.backward(retain_graph=True)

            for name, module in self.model.named_modules():
                if isinstance(module, BatchEnsembleMaskedWrapper):
                    grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                    for i, g in enumerate(grad):
                        grads[i][name].append(torch.abs(g).cpu())

        self.model.zero_grad()

        remove_wrappers_from_model(self.model)

        for mi, ens_grads in grads.items():
            f = lambda x: torch.mean(x, 0)

            if self.grad_reduce == 'median':
                f = lambda x: torch.median(x, 0)
            elif self.grad_reduce == 'max':
                f = lambda x: torch.max(x, 0)

            ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in ens_grads.items()}

            masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=self.prune_percentage,
                                             global_pruning=self.global_pruning, device=self.device)

            model = extract_inner_model(self.model, masks, re_init=self.re_init)

            p = calculate_trainable_parameters(model)
            print(mi, p)

            self.models.append(model)

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
            m = torch.load(os.path.join(path, 'model_{}.pt'.format(i)), map_location=self.device)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        for i, m in enumerate(self.models):
            torch.save(m, os.path.join(path, 'model_{}.pt'.format(i)))


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

            # print(self.model)

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

                # classifier for vgg
                # fc for ResNet
                # print(ens_grads['fc'])
                
                masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=pruning,
                                                 global_pruning=self.global_pruning, device=self.device)

                model = extract_inner_model(self.model, masks, re_init=self.re_init)

                if last_iteration:
                    self.models.append(model)
                else:
                    self.model = model

                p = calculate_trainable_parameters(model)
                print(mi, last_iteration, p, calculate_trainable_parameters(self.model))

        all_scores = []

        for i in tqdm(range(len(self.models)), desc='Training models'):
            model = self.models[i]

            # print(model)

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
            # model.to('cpu')
            all_scores.append(scores)
        # self.device = 'cpu'
        return all_scores

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        outputs = torch.stack([m(x) for m in self.models], 1)
        if reduce:
            outputs = torch.mean(outputs, 1)
        return outputs

    def load(self, path):
        # self.device = 'cpu'
        for i in range(self.ensemble):
            with open(os.path.join(path, 'model_{}.pt'.format(i)), 'rb') as file:
                # m = pickle.load(file)
                m = torch.load(os.path.join(path, 'model_{}.pt'.format(i)),
                               map_location=self.device)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        for i, m in enumerate(self.models):
            with open(os.path.join(path, 'model_{}.pt'.format(i)), 'wb') as file:
                pickle.dump(m, file)
            # torch.save(m, os.path.join(path, 'model_{}.pt'.format(i)))