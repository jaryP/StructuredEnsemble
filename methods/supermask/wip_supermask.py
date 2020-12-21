import os
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from eval import eval_model
from methods.base import EnsembleMethod
from methods.supermask.base import layer_to_masked, mask_training, extract_distribution_subnetwork, be_model_training
from methods.supermask.layers import EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper
from methods.supermask.trainable_masks import MMD
from utils import train_model


class ReverseSuperMask(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):

        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.prune_percentage = method_parameters['prune_percentage']
        self.ensemble = method_parameters['n_ensemble']
        self.re_init = method_parameters['re_init']
        self.global_pruning = method_parameters['global_pruning']
        self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']

        self.model = deepcopy(model)
        self.models = torch.nn.ModuleList()
        self.distributions = torch.nn.ModuleDict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        optim = optimizer([param for name, param in self.model.named_parameters()
                           if param.requires_grad and 'distributions' not in name])

        train_scheduler = scheduler(optim)

        best_model, scores, best_model_scores, losses = train_model(model=self.model, optimizer=optim,
                                                                    epochs=epochs, train_loader=train_dataset,
                                                                    scheduler=train_scheduler,
                                                                    early_stopping=early_stopping,
                                                                    test_loader=test_dataset, eval_loader=eval_dataset,
                                                                    device=self.device)

        self.model.load_state_dict(best_model)
        print(best_model_scores)

        layer_to_masked(self.model, masks_params=self.supermask_parameters, ensemble=self.ensemble)
        mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
                      dataset=train_dataset, ensemble=self.ensemble)

        for name, module in self.model.named_modules():
            if isinstance(module, EnsembleMaskedWrapper):
                distr = module.distributions
                self.distributions[name] = distr

        for i in tqdm(range(self.ensemble)):
            m = extract_distribution_subnetwork(self.model, self.prune_percentage, i,
                                                re_init=self.re_init, global_pruning=self.global_pruning)

            optim = optimizer([param for name, param in m.named_parameters()
                               if param.requires_grad and 'distributions' not in name])

            train_scheduler = scheduler(optim)

            best_model, scores, best_model_scores, losses = train_model(model=m, optimizer=optim,
                                                                        epochs=5, train_loader=train_dataset,
                                                                        scheduler=train_scheduler,
                                                                        early_stopping=early_stopping,
                                                                        test_loader=test_dataset,
                                                                        eval_loader=eval_dataset,
                                                                        device=self.device)
            m.load_state_dict(best_model)
            self.models.append(m)

        all_scores = []

        # for i in tqdm(range(len(self.models)), desc='Training models'):
        #     model = self.models[i]
        #     optim = optimizer([param for name, param in model.named_parameters() if param.requires_grad])
        #     train_scheduler = scheduler(optim)
        #
        #     best_model, scores, best_model_scores = train_model(model=model, optimizer=optim,
        #                                                         epochs=epochs, train_loader=train_dataset,
        #                                                         scheduler=train_scheduler,
        #                                                         early_stopping=early_stopping,
        #                                                         test_loader=test_dataset, eval_loader=eval_dataset,
        #                                                         device=self.device)
        #
        #     model.load_state_dict(best_model)
        #
        #     all_scores.append(scores)

        return all_scores

    def predict_proba(self, x, y, **kwargs):
        x, y = x.to(self.device), y.to(self.device)
        outputs = torch.stack([m(x) for m in self.models])
        outputs = torch.mean(outputs, 0)
        outputs = torch.nn.functional.softmax(outputs, -1)

        return outputs

    def load(self, path):
        self.distributions = torch.load(os.path.join(path, 'distributions.pt'), map_location=self.device)
        for i in range(self.ensemble):
            m = torch.load(os.path.join(path, 'model_{}.pt'.format(i)), map_location=self.device)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        torch.save(self.distributions, os.path.join(path, 'distributions.pt'))
        for i, m in enumerate(self.models):
            torch.save(m, os.path.join(path, 'model_{}.pt'.format(i)))


class BatchPruningSuperMask(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.ensemble = method_parameters['n_ensemble']
        # self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']
        # self.train_before = method_parameters.get('train_before', True)

        self.model = deepcopy(model)
        self.distributions = torch.nn.ModuleDict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):
        layer_to_masked(self.model, masks_params=self.supermask_parameters, ensemble=self.ensemble, batch_ensemble=True)

        # mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
        #               dataset=train_dataset, ensemble=self.ensemble)
        #
        # for name, module in self.model.named_modules():
        #     if isinstance(module, EnsembleMaskedWrapper):
        #         distr = module.distributions
        #         self.distributions[name] = distr

        optim = optimizer([param for name, param in self.model.named_parameters()
                           if param.requires_grad])

        train_scheduler = scheduler(optim)

        best_model, scores, best_model_scores, losses = be_model_training(model=self.model, optimizer=optim,
                                                                          epochs=epochs, train_loader=train_dataset,
                                                                          scheduler=train_scheduler,
                                                                          early_stopping=early_stopping,
                                                                          test_loader=test_dataset,
                                                                          eval_loader=eval_dataset,
                                                                          device=self.device)

        self.model.load_state_dict(best_model)

        """
        grads = defaultdict(lambda: defaultdict(list))

        for i, (x, y) in enumerate(train_dataset):
            # if i == 1:
            #     break
            x = torch.cat([x for _ in range(self.ensemble)], dim=0)
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)
            loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

            self.model.zero_grad()
            loss.backward(retain_graph=True)

            for name, module in self.model.named_modules():
                if isinstance(module, BatchEnsembleMaskedWrapper):
                    grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                    # assert len(grad) == 1
                    grad = grad[0]
                    # grad = module.last_mask.grad
                    # params = [p.grad.detach() for name, p in module.named_parameters()
                    #           if 'distributions.{}'.format(ens) in name and p.grad is not None]
                    # assert len(params) == 1
                    for i, g in enumerate(grad):
                        grads[i][name].append(torch.abs(g).cpu())

        self.model.zero_grad()

        for i, d in grads.items():
            grads = {name: torch.mean(torch.stack(gs, 0), 0).detach().cpu() for name, gs in d.items()}

            if self.global_pruning:
                stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in grads.items()])
                grads_sum = np.sum(stacked_grads)
                stacked_grads = stacked_grads / grads_sum

                threshold = np.quantile(stacked_grads, q=self.prune_percentage)

                grads = {name: gs / grads_sum for name, gs in grads.items()}

                masks = {name: torch.ge(gs, threshold).float().to(self.device)
                         for name, gs in grads.items()}
            else:
                masks = {name: torch.ge(gs, torch.quantile(gs, self.prune_percentage)).float()
                         for name, gs in grads.items()}

            model = deepcopy(self.model)

            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    mask = masks[name].squeeze()

                    if mask.sum() == 0:
                        max = torch.argmax(grads[name])
                        mask = torch.zeros_like(mask)
                        mask[max] = 1.0

                    module.layer.register_buffer('mask', mask)
            """
        # if not self.train_before:
        #     mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
        #                   dataset=train_dataset, ensemble=self.ensemble)

        return [scores]

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        bs = x.shape[0]
        x = torch.cat([x for _ in range(self.ensemble)], dim=0)
        outputs = self.model(x)
        outputs = outputs.view([self.ensemble, bs, -1])
        if reduce:
            outputs = torch.mean(outputs, 0)
        return outputs

    def load(self, path):
        self.distributions = torch.load(os.path.join(path, 'distributions.pt'), map_location=self.device)
        self.model = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        self.model.to(self.device)

    def save(self, path):
        torch.save(self.distributions, os.path.join(path, 'distributions.pt'))
        torch.save(self.model, os.path.join(path, 'model.pt'))


class BatchSuperMask(EnsembleMethod):
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

        self.model = deepcopy(model)
        self.distributions = torch.nn.ModuleDict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        layer_to_masked(self.model, masks_params=self.supermask_parameters, ensemble=self.ensemble, batch_ensemble=True)

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

        optim = optimizer([param for name, param in self.model.named_parameters()
                           if param.requires_grad and 'distributions' not in name])

        train_scheduler = scheduler(optim)

        best_model, scores, best_model_scores, losses = be_model_training(model=self.model, optimizer=optim,
                                                                          epochs=epochs,
                                                                          train_loader=train_dataset,
                                                                          scheduler=train_scheduler,
                                                                          early_stopping=early_stopping,
                                                                          test_loader=test_dataset,
                                                                          eval_loader=eval_dataset,
                                                                          device=self.device, w=0)

        self.model.load_state_dict(best_model)

        """
        grads = defaultdict(lambda: defaultdict(list))
        
        for i, (x, y) in enumerate(train_dataset):
            # if i == 1:
            #     break
            x = torch.cat([x for _ in range(self.ensemble)], dim=0)
            x, y = x.to(self.device), y.to(self.device)
            
            pred = self.model(x)
            loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

            self.model.zero_grad()
            loss.backward(retain_graph=True)
            
            for name, module in self.model.named_modules():
                if isinstance(module, BatchEnsembleMaskedWrapper):
                    grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                    # assert len(grad) == 1
                    grad = grad[0]
                    # grad = module.last_mask.grad
                    # params = [p.grad.detach() for name, p in module.named_parameters()
                    #           if 'distributions.{}'.format(ens) in name and p.grad is not None]
                    # assert len(params) == 1
                    for i, g in enumerate(grad):
                        grads[i][name].append(torch.abs(g).cpu())

        self.model.zero_grad()
        
        for i, d in grads.items():
            grads = {name: torch.mean(torch.stack(gs, 0), 0).detach().cpu() for name, gs in d.items()}

            if self.global_pruning:
                stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in grads.items()])
                grads_sum = np.sum(stacked_grads)
                stacked_grads = stacked_grads / grads_sum

                threshold = np.quantile(stacked_grads, q=self.prune_percentage)

                grads = {name: gs / grads_sum for name, gs in grads.items()}

                masks = {name: torch.ge(gs, threshold).float().to(self.device)
                         for name, gs in grads.items()}
            else:
                masks = {name: torch.ge(gs, torch.quantile(gs, self.prune_percentage)).float()
                         for name, gs in grads.items()}
            
            model = deepcopy(self.model)

            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    mask = masks[name].squeeze()

                    if mask.sum() == 0:
                        max = torch.argmax(grads[name])
                        mask = torch.zeros_like(mask)
                        mask[max] = 1.0

                    module.layer.register_buffer('mask', mask)
            """

        return [scores]

    def predict_logits(self, x, y, reduce):
        x, y = x.to(self.device), y.to(self.device)
        bs = x.shape[0]
        x = torch.cat([x for _ in range(self.ensemble)], dim=0)
        outputs = self.model(x)
        outputs = outputs.view([self.ensemble, bs, -1])
        if reduce:
            outputs = torch.mean(outputs, 0)
        return outputs

    def load(self, path):
        self.distributions = torch.load(os.path.join(path, 'distributions.pt'), map_location=self.device)
        self.model = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        self.model.to(self.device)

    def save(self, path):
        torch.save(self.distributions, os.path.join(path, 'distributions.pt'))
        torch.save(self.model, os.path.join(path, 'model.pt'))
