import os
from collections import defaultdict
from copy import deepcopy

import torch
from tqdm import tqdm
import numpy as np

from methods.base import EnsembleMethod
from methods.supermask.base import layer_to_masked, extract_distribution_subnetwork, mask_training, \
    iterative_mask_training, masked_to_layer, extract_structured, get_mask
from methods.supermask.layers import EnsembleMaskedWrapper
from utils import train_model, calculate_trainable_parameters

import matplotlib.pyplot as plt
import seaborn as sns


# TODO: implementare estrazione delle sottoreti da una singola distribuzione
# TODO: implementare pruning utilizzando il gradiente delle maschere

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


class GradSuperMask(EnsembleMethod):
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

        self.model.cpu()

        for ens in range(self.ensemble):
            model = deepcopy(self.model).to(self.device)

            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    module.set_distribution(ens)

            grads = defaultdict(list)

            for i, (x, y) in enumerate(train_dataset):
                # if i == 1:
                #     break
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

                model.zero_grad()
                loss.backward(retain_graph=True)

                for name, module in model.named_modules():
                    if isinstance(module, EnsembleMaskedWrapper):
                        grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
                        assert len(grad) == 1
                        grad = grad[0]
                        # grad = module.last_mask.grad
                        # params = [p.grad.detach() for name, p in module.named_parameters()
                        #           if 'distributions.{}'.format(ens) in name and p.grad is not None]
                        # assert len(params) == 1
                        grads[name].append(torch.abs(grad).cpu())

            model.zero_grad()

            grads = {name: torch.mean(torch.stack(gs, 0), 0).detach().cpu() for name, gs in grads.items()}

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

            # m = deepcopy(model)
            #
            # masks = {}

            # for name, module in model.named_modules():
            #     if isinstance(module, EnsembleMaskedWrapper):
            #         mask = get_mask(module, self.prune_percentage, ens, threshold=None).squeeze()
            #         module.layer.register_buffer('mask', mask)
            #         # masks[name] = mask

            # masks = register_masks(m, prune_percentage, distribution, global_pruning=global_pruning)
            # extract_structured(m, re_init=re_init)

            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    mask = masks[name].squeeze()

                    if mask.sum() == 0:
                        max = torch.argmax(grads[name])
                        mask = torch.zeros_like(mask)
                        mask[0, max] = 1.0

                    module.layer.register_buffer('mask', mask)

            model = extract_structured(masked_to_layer(model), re_init=self.re_init)

            p = calculate_trainable_parameters(model)
            print(ens, p)

            self.models.append(model)

        # input()
        # for name, module in self.model.named_modules():
        #     if isinstance(module, EnsembleMaskedWrapper):
        #         distr = module.distributions
        #         self.final_distributions[name] = (deepcopy(distr), deepcopy(module.layer.weight))
        #
        # grads_sum = torch.cat([torch.flatten(param.grad) for name, param in self.model.named_parameters()
        #                    if 'distributions.0' in name]).sum()
        #
        # grads_dict = {name[0]: torch.flatten(param.grad) / grads_sum for name, param in self.model.named_parameters()
        #               if 'distributions.0' in name}
        #
        # print(grads_dict.keys())
        # print(self.final_distributions.keys())
        #
        # # threshold = np.quantile(grads / grads_sum, q=prune_percentage)
        #
        # # print(grads)
        # #
        # # for n, m in self.model.named_modules():
        # #     if isinstance(m, EnsembleMaskedWrapper):
        # #         grads_dict[n] = m.mask.grad
        # #
        # # for k, (v, w) in self.initial_distributions.items():
        # #     plt.figure(k)
        # #     ax = plt.subplot(121)
        # #     d = v[0](size=5, reduce=False)
        # #     d = d.squeeze().detach().cpu().numpy()
        # #     # sns.barplot(data=d, ax=ax)
        # #     ax.bar(range(len(d)), d)
        # #
        # #     ax = plt.subplot(122, sharey=ax)
        # #     d = v[1](size=5, reduce=False)
        # #     d = d.squeeze().detach().cpu().numpy()
        # #     # sns.barplot(data=d, ax=ax)
        # #     ax.bar(range(len(d)), d)
        # #
        # # plt.show()
        #
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
        #     plt.figure('grad_{}'.format(k))
        #     ax = plt.subplot(111)
        #     grad = grads_dict[k].abs().squeeze().detach().cpu().numpy()
        #     ax.bar(range(len(grad)), grad)
        #
        #     # w = .squeeze().detach().cpu().numpy()
        #
        # plt.show()
        #
        # # _, final_loss = eval_model(self.model, dataset=train_dataset, device=self.device)
        # # model_params = calculate_trainable_parameters(self.model)
        # #
        # # pps = []
        # #
        # # for i in range(self.ensemble):
        # #     best_loss = numpy.inf
        # #     best_pp = 0
        # #     for pp in numpy.linspace(0.1, 0.9, 20):
        # #         m, masks = extract_distribution_subnetwork(self.model, pp, 0,
        # #                                                    re_init=self.re_init, global_pruning=self.global_pruning)
        # #         _, loss = eval_model(m, dataset=train_dataset, device=self.device)
        # #
        # #         # p = calculate_trainable_parameters(m)
        # #         # pruning = p / model_params
        # #         # loss_diff = abs(final_loss - loss)
        # #
        # #         if loss < best_loss:
        # #             best_loss = loss
        # #             best_pp = pp
        # #
        # #         print(pp, loss, best_pp, best_loss)
        # #         # return pruning + loss_diff
        # #     pps.append(best_pp)
        # # # result = optimize.minimize(f, [0.5])
        #
        # for i in range(self.ensemble):
        #
        #     m, masks = extract_distribution_subnetwork(self.model, self.prune_percentage, i,
        #                                                re_init=self.re_init, global_pruning=self.global_pruning)
        #
        #     self.models.append(m)
        #
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


class _GradSuperMask(EnsembleMethod):
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

        class SaveOutputs:
            def __init__(self, m, name):
                self.output = None
                self.module_name = name
                self.handle = m.register_forward_hook(self.hook_fn)

            def hook_fn(self, module, input, output):
                self.output = output

            def remove(self):
                self.handle.remove()

        layer_to_masked(self.model, masks_params=self.supermask_parameters,
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

        self.model.cpu()

        for ens in range(self.ensemble):
            model = deepcopy(self.model).to(self.device)

            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    module.set_distribution(ens)
                    h = SaveOutputs(module, name)
                    hooks.append(h)

            grads = defaultdict(list)

            for i, (x, y) in enumerate(train_dataset):
                if i == 1:
                    break
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

                model.zero_grad()
                loss.backward()

                for h in hooks:
                    grad = h.output
                    # print(len(grad), grad[0].shape)
                    name = h.module_name
                    # for name, module in model.named_modules():
                    #     if isinstance(module, EnsembleMaskedWrapper):
                    #         params = [p.grad.detach() for name, p in module.named_parameters()
                    #                   if 'distributions.{}'.format(ens) in name and p.grad is not None]
                    #         assert len(params) == 1
                    grads[name].append(torch.abs(grad).cpu())

            model.zero_grad()

            grads = {name: torch.mean(torch.stack(gs, 0), 0).detach().cpu() for name, gs in grads.items()}
            thresholds = {name: np.quantile(g.view(-1).numpy(), self.prune_percentage) for name, g in grads.items()}

            stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in grads.items()])
            grads_sum = np.sum(stacked_grads)
            stacked_grads = stacked_grads / grads_sum

            threshold = np.quantile(stacked_grads, q=self.prune_percentage)

            masks = {name: torch.ge(gs / grads_sum, threshold).float().to(self.device) for name, gs in grads.items()}

            for name, module in model.named_modules():
                if isinstance(module, EnsembleMaskedWrapper):
                    module.layer.register_buffer('mask', masks[name])

            model = extract_structured(masked_to_layer(model), re_init=self.re_init)

            p = calculate_trainable_parameters(model)
            print(ens, p)

            self.models.append(model)

        # input()
        # for name, module in self.model.named_modules():
        #     if isinstance(module, EnsembleMaskedWrapper):
        #         distr = module.distributions
        #         self.final_distributions[name] = (deepcopy(distr), deepcopy(module.layer.weight))
        #
        # grads_sum = torch.cat([torch.flatten(param.grad) for name, param in self.model.named_parameters()
        #                    if 'distributions.0' in name]).sum()
        #
        # grads_dict = {name[0]: torch.flatten(param.grad) / grads_sum for name, param in self.model.named_parameters()
        #               if 'distributions.0' in name}
        #
        # print(grads_dict.keys())
        # print(self.final_distributions.keys())
        #
        # # threshold = np.quantile(grads / grads_sum, q=prune_percentage)
        #
        # # print(grads)
        # #
        # # for n, m in self.model.named_modules():
        # #     if isinstance(m, EnsembleMaskedWrapper):
        # #         grads_dict[n] = m.mask.grad
        # #
        # # for k, (v, w) in self.initial_distributions.items():
        # #     plt.figure(k)
        # #     ax = plt.subplot(121)
        # #     d = v[0](size=5, reduce=False)
        # #     d = d.squeeze().detach().cpu().numpy()
        # #     # sns.barplot(data=d, ax=ax)
        # #     ax.bar(range(len(d)), d)
        # #
        # #     ax = plt.subplot(122, sharey=ax)
        # #     d = v[1](size=5, reduce=False)
        # #     d = d.squeeze().detach().cpu().numpy()
        # #     # sns.barplot(data=d, ax=ax)
        # #     ax.bar(range(len(d)), d)
        # #
        # # plt.show()
        #
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
        #     plt.figure('grad_{}'.format(k))
        #     ax = plt.subplot(111)
        #     grad = grads_dict[k].abs().squeeze().detach().cpu().numpy()
        #     ax.bar(range(len(grad)), grad)
        #
        #     # w = .squeeze().detach().cpu().numpy()
        #
        # plt.show()
        #
        # # _, final_loss = eval_model(self.model, dataset=train_dataset, device=self.device)
        # # model_params = calculate_trainable_parameters(self.model)
        # #
        # # pps = []
        # #
        # # for i in range(self.ensemble):
        # #     best_loss = numpy.inf
        # #     best_pp = 0
        # #     for pp in numpy.linspace(0.1, 0.9, 20):
        # #         m, masks = extract_distribution_subnetwork(self.model, pp, 0,
        # #                                                    re_init=self.re_init, global_pruning=self.global_pruning)
        # #         _, loss = eval_model(m, dataset=train_dataset, device=self.device)
        # #
        # #         # p = calculate_trainable_parameters(m)
        # #         # pruning = p / model_params
        # #         # loss_diff = abs(final_loss - loss)
        # #
        # #         if loss < best_loss:
        # #             best_loss = loss
        # #             best_pp = pp
        # #
        # #         print(pp, loss, best_pp, best_loss)
        # #         # return pruning + loss_diff
        # #     pps.append(best_pp)
        # # # result = optimize.minimize(f, [0.5])
        #
        # for i in range(self.ensemble):
        #
        #     m, masks = extract_distribution_subnetwork(self.model, self.prune_percentage, i,
        #                                                re_init=self.re_init, global_pruning=self.global_pruning)
        #
        #     self.models.append(m)
        #
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
            print(best_model_scores)

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

