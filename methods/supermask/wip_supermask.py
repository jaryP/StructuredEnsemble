import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from tqdm import tqdm

from eval import eval_model
from methods.base import EnsembleMethod
from methods.supermask.base import layer_to_masked, mask_training, extract_distribution_subnetwork, be_model_training, \
    iterative_mask_training, masked_to_layer
from methods.supermask.layers import EnsembleMaskedWrapper
from methods.supermask.models_utils import extract_inner_model, add_wrappers_to_model
from utils import train_model, calculate_trainable_parameters


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

        add_wrappers_to_model(self.model, masks_params=self.supermask_parameters,
                              ensemble=self.ensemble, single_distribution=self.single_distribution)

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


class SuperMask(EnsembleMethod):
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

        self.iterative_training = method_parameters['iterative_training']
        self.divergence = method_parameters.get('divergence', 'mmd')
        self.divergence_w = method_parameters.get('divergence_w', 1)

        self.single_distribution = method_parameters.get('single_distribution', False)

        self.model = deepcopy(model)
        self.models = torch.nn.ModuleList()
        self.final_distributions = dict()
        self.initial_distributions = dict()

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        layer_to_masked(self.model, masks_params=self.supermask_parameters,
                        ensemble=self.ensemble, single_distribution=self.single_distribution)

        # for name, module in self.model.named_modules():
        #     if isinstance(module, EnsembleMaskedWrapper):
        #         distr = module.distributions
        #         self.initial_distributions[name] = deepcopy(distr)

        if self.iterative_training and not self.single_distribution:
            iterative_mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
                                    dataset=train_dataset, ensemble=self.ensemble,
                                    divergence_type=self.divergence, w=self.divergence_w)
        else:
            mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
                          dataset=train_dataset, ensemble=self.ensemble, w=self.divergence_w)

        # for name, module in self.model.named_modules():
        #     if isinstance(module, EnsembleMaskedWrapper):
        #         distr = module.distributions
        #         self.final_distributions[name] = (deepcopy(distr), deepcopy(module.weight))

        # for k, (v, w) in self.initial_distributions.items():
        #     plt.figure(k)
        #     ax = plt.subplot(121)
        #     d = v[0](size=5, reduce=False)
        #     d = d.squeeze().detach().cpu().numpy()
        #     # sns.barplot(data=d, ax=ax)
        #     ax.bar(range(len(d)), d)
        #
        #     ax = plt.subplot(122, sharey=ax)
        #     d = v[1](size=5, reduce=False)
        #     d = d.squeeze().detach().cpu().numpy()
        #     # sns.barplot(data=d, ax=ax)
        #     ax.bar(range(len(d)), d)
        #
        # plt.show()

        # for k, (v, w) in self.final_distributions.items():
        #     plt.figure(k)
        #     ax = plt.subplot(121)
        #     d = v[0](size=5, reduce=False)
        #     d = d.squeeze().detach().cpu().numpy()
        #     # sns.barplot(data=d, ax=ax)
        #     ax.bar(range(len(d)), d)
        #
        #     ax = plt.subplot(122, sharey=ax)
        #     d = v[1](size=5, reduce=False)
        #     d = d.squeeze().detach().cpu().numpy()
        #     # sns.barplot(data=d, ax=ax)
        #     ax.bar(range(len(d)), d)
        #
        #     # plt.figure('module_{}'.format(k))
        #     # w = .squeeze().detach().cpu().numpy()

        # plt.show()

        # _, final_loss = eval_model(self.model, dataset=train_dataset, device=self.device)
        # model_params = calculate_trainable_parameters(self.model)
        #
        # pps = []
        #
        # for i in range(self.ensemble):
        #     best_loss = numpy.inf
        #     best_pp = 0
        #     for pp in numpy.linspace(0.1, 0.9, 20):
        #         m, masks = extract_distribution_subnetwork(self.model, pp, 0,
        #                                                    re_init=self.re_init, global_pruning=self.global_pruning)
        #         _, loss = eval_model(m, dataset=train_dataset, device=self.device)
        #
        #         # p = calculate_trainable_parameters(m)
        #         # pruning = p / model_params
        #         # loss_diff = abs(final_loss - loss)
        #
        #         if loss < best_loss:
        #             best_loss = loss
        #             best_pp = pp
        #
        #         print(pp, loss, best_pp, best_loss)
        #         # return pruning + loss_diff
        #     pps.append(best_pp)
        # # result = optimize.minimize(f, [0.5])

        for i in range(self.ensemble):
            m, masks = extract_distribution_subnetwork(self.model, self.prune_percentage, i,
                                                       re_init=self.re_init, global_pruning=self.global_pruning)

            self.models.append(m)

        #     for k, v in masks.items():
        #         v = v.cpu().numpy()
        #         name = str(k)
        #         f = plt.figure(name)
        #         plt.suptitle(name, y=1.05, fontsize=18)
        #         ax = f.add_subplot(1, self.ensemble, i + 1)
        #         ax.bar(range(len(v)), v)
        #
        # plt.show()

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

    # def predict_proba(self, x, y, **kwargs):
    #     x, y = x.to(self.device), y.to(self.device)
    #     outputs = torch.stack([m(x) for m in self.models])
    #     outputs = torch.mean(outputs, 0)
    #     outputs = torch.nn.functional.softmax(outputs, -1)
    #
    #     return outputs

    def load(self, path):
        self.final_distributions = torch.load(os.path.join(path, 'final_distributions.pt'), map_location=self.device)
        self.initial_distributions = torch.load(os.path.join(path, 'initial_distributions.pt'),
                                                map_location=self.device)

        for k, v in self.final_distributions.items():
            plt.figure(k)
            ax = plt.subplot(121)
            d = v[0](size=5, reduce=False)
            d = d.squeeze().detach().cpu().numpy()
            sns.barplot(data=d, ax=ax)

            ax = plt.subplot(122, sharey=ax)
            d = v[1](size=5, reduce=False)
            d = d.squeeze().detach().cpu().numpy()
            sns.barplot(data=d, ax=ax)

        plt.show()

        for i in range(self.ensemble):
            m = torch.load(os.path.join(path, 'model_{}.pt'.format(i)), map_location=self.device)
            m.to(self.device)
            self.models.append(m)

    def save(self, path):
        torch.save(self.final_distributions, os.path.join(path, 'final_distributions.pt'))
        torch.save(self.initial_distributions, os.path.join(path, 'initial_distributions.pt'))

        for i, m in enumerate(self.models):
            torch.save(m, os.path.join(path, 'model_{}.pt'.format(i)))


class TreeSuperMask(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):

        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device
        self.prune_percentage = method_parameters['prune_percentage']
        self.global_pruning = method_parameters['global_pruning']

        self.ensemble = method_parameters['n_ensemble']

        self.re_init_first = method_parameters['re_init_first']
        self.re_init_second = method_parameters['re_init_second']
        self.drop_last = method_parameters['drop_last']

        self.mask_epochs = method_parameters['mask_epochs']
        self.supermask_parameters = method_parameters['supermask']
        self.shrink_iterations = method_parameters.get('shrink_iterations', 0)

        self.grad_reduce = method_parameters.get('grad_reduce', 'mean')

        # self.iterative_training = method_parameters['iterative_training']
        # self.divergence = method_parameters.get('divergence', 'mmd')
        # self.divergence_w = method_parameters.get('divergence_w', 1)
        #
        # self.single_distribution = method_parameters.get('single_distribution', False)

        self.model = deepcopy(model)
        self.models = torch.nn.ModuleList()
        self.final_distributions = dict()
        self.initial_distributions = dict()

        if self.prune_percentage == 'adaptive':
            self.prune_percentage = 1 - (1 / self.ensemble)

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
                     regularization=None, early_stopping=None,
                     **kwargs):

        for i in tqdm(range(self.ensemble)):

            if i == self.ensemble - 1 and not self.drop_last:
                self.models.append(self.model)
                continue

            layer_to_masked(self.model, masks_params=self.supermask_parameters,
                            ensemble=1, single_distribution=False)

            mask_training(epochs=self.mask_epochs, model=self.model, device=self.device,
                          dataset=train_dataset, ensemble=1)

            train_scores, _ = eval_model(self.model, train_dataset, device=self.device)
            print(train_scores)

            grads = defaultdict(list)

            for _, (x, y) in enumerate(train_dataset):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                for name, module in self.model.named_modules():
                    if isinstance(module, EnsembleMaskedWrapper):
                        grad = torch.autograd.grad(loss, module.last_mask[i], retain_graph=True)
                        assert len(grad) == 1
                        grad = grad[0]
                        # grad = module.last_mask.grad
                        # params = [p.grad.detach() for name, p in module.named_parameters()
                        #           if 'distributions.{}'.format(ens) in name and p.grad is not None]
                        # assert len(params) == 1
                        grads[name].append(torch.abs(grad).cpu())

            self.model.zero_grad()

            f = lambda x: torch.mean(x, 0)

            if self.grad_reduce == 'median':
                f = lambda x: torch.median(x, 0)
            elif self.grad_reduce == 'max':
                f = lambda x: torch.max(x, 0)

            grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in grads.items()}

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

            for name, mask in masks.items():
                mask = mask.squeeze()
                if mask.sum() == 0:
                    max = torch.argmax(grads[name])
                    mask = torch.zeros_like(mask)
                    mask[max] = 1.0
                masks[name] = mask

            masks2 = {name: 1 - m for name, m in masks.items()}

            masked_to_layer(self.model)
            po = calculate_trainable_parameters(self.model)
            model_1 = extract_inner_model(self.model, masks, re_init=self.re_init_first)
            p1 = calculate_trainable_parameters(model_1)
            model_2 = extract_inner_model(self.model, masks2, re_init=self.re_init_second)
            p2 = calculate_trainable_parameters(model_2)

            print(po, p1, p2)

            self.model = deepcopy(model_1)
            self.models.append(deepcopy(model_1))

        print('all params', sum([calculate_trainable_parameters(m) for m in self.models]))

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
        #
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