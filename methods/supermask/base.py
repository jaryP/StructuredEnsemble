from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch import nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.models import VGG
from tqdm import tqdm

from eval import eval_model
from .layers import EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper
from .trainable_masks import MMD


def mask_training(model, epochs, dataset, ensemble, device='cpu', parameters=None, w=1):
    model.to(device)

    bar = tqdm(range(epochs), desc='Mask: {}'.format('Mask training'), leave=False)

    if parameters is None:
        parameters = [param for name, param in model.named_parameters() if 'distributions' in name]

    optim = torch.optim.Adam(parameters, lr=0.001)

    for e in bar:
        losses = []
        model.train()

        for i, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

            kl = torch.tensor(0.0, device=device)
            entropy = torch.tensor(0.0, device=device)

            div = 0

            if ensemble > 1:
                for name, module in model.named_modules():
                    if isinstance(module, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
                        distr = module.distributions
                        _mmd = torch.tensor(0.0, device=device)
                        div += len(distr)
                        for i, d1 in enumerate(distr):
                            for j in range(i + 1, len(distr)):
                                _mmd += MMD(d1, distr[j])
                                e = d1.posterior.entropy()
                                entropy += e.mean()
                        kl += _mmd

                kl = kl / div
                kl = 1 / kl
                kl *= w
                entropy = entropy / div

            losses.append((loss.item(), kl.item(), entropy.item()))

            loss += kl + entropy

            optim.zero_grad()
            loss.backward()
            optim.step()

        bar.set_postfix({'losses': np.mean(losses, 0)})


def iterative_mask_training(model, epochs, dataset, ensemble, device='cpu', parameters=None, w=1,
                            divergence_type='global_mmd'):

    def divergence(current_ens):
        d = torch.tensor(0.0, device=device)
        all_distrs = defaultdict(list)

        for name, module in model.named_modules():
            if isinstance(module, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
                distr = module.distributions
                d1 = distr[current_ens]
                if divergence_type == 'mmd':
                    for j in range(current_ens):
                        d += MMD(d1, distr[j])
                elif divergence_type == 'global_mmd':
                    all_distrs[current_ens].append(d1().squeeze())
                    for j in range(current_ens):
                        all_distrs[j].append(distr[j]().squeeze())
                else:
                    assert False

            # all_distrs = defaultdict(list)
            # for name, module in model.named_modules():
            #     if isinstance(module, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
            #         distr = module.distributions
            #         all_distrs[current_ens].append(distr[current_ens]().squeeze())
            #         for j in range(current_ens):
            #             all_distrs[j].append(distr[j]().squeeze())

        if divergence_type == 'global_mmd':
            all_distrs = {k: torch.cat(v, 0) for k, v in all_distrs.items()}
            current_d = all_distrs[current_ens]
            for k, v in all_distrs.items():
                if k == current_ens:
                    continue
                d += MMD(current_d, v)

        return d

    model.to(device)

    distributions = dict()

    for name, module in model.named_modules():
        if isinstance(module, EnsembleMaskedWrapper):
            distr = module.distributions
            distributions[name] = distr

    for ens in tqdm(range(ensemble)):
        if parameters is None:
            pp = [param for name, param in model.named_parameters() if 'distributions.{}'.format(ens) in name]
        else:
            pp = parameters

        optim = torch.optim.Adam(pp, lr=0.001)

        for name, module in model.named_modules():
            if isinstance(module, EnsembleMaskedWrapper):
                module.set_distribution(ens)

        bar = tqdm(range(epochs), desc='Mask training', leave=False)

        for _ in bar:
            losses = []
            model.train()

            for i, (x, y) in enumerate(dataset):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = torch.nn.functional.cross_entropy(pred, y, reduction='mean')

                kl = torch.tensor(0.0, device=device)

                if ens > 0:
                    kl = divergence(ens)
                    kl = 1 / (kl + 1e-12)
                    kl *= w

                losses.append((loss.item(), kl.item()))

                loss += kl

                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()

            bar.set_postfix({'losses': np.mean(losses, 0)})


def be_model_training(model, optimizer, train_loader, epochs, scheduler, early_stopping=None,
                      test_loader=None, eval_loader=None, device='cpu', w=1):

    def divergence():
        d = torch.tensor(0.0, device=device)

        for name, module in model.named_modules():
            if isinstance(module, (EnsembleMaskedWrapper,
                                   BatchEnsembleMaskedWrapper)):
                distr = module.distributions
                for i, d1 in enumerate(distr):
                    for j, d2 in enumerate(distr):
                        if j <= i:
                            continue
                        d += MMD(d1, d2)

        return d

    model.to(device)

    distributions = dict()

    for name, module in model.named_modules():
        if isinstance(module, EnsembleMaskedWrapper):
            distr = module.distributions
            distributions[name] = distr

    for name, module in model.named_modules():
        if isinstance(module, EnsembleMaskedWrapper):
            module.set_distribution('all')

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0
    model.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True, desc='Mask training')

    for epoch in bar:
        model.train()
        losses = []
        kl_losses = []

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = torch.nn.functional.cross_entropy(pred, y, reduction='none')
            losses.extend(loss.tolist())
            loss = loss.mean()

            kl = divergence()
            kl = 1 / (kl + 1e-12)
            kl *= w
            kl_losses.append(kl.item())
            loss += kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()

        if eval_loader is not None:
            eval_scores, _ = eval_model(model, eval_loader, topk=[1, 5], device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_model_i = epoch

        train_scores, _ = eval_model(model, train_loader, device=device)
        test_scores, _ = eval_model(model, test_loader, device=device)

        kl_losses = sum(kl_losses) / len(kl_losses)
        bar.set_postfix({'Train score': train_scores[1], 'Test score': test_scores[1],
                         'Eval score': eval_scores[1] if eval_scores != 0 else 0,
                         'Mean loss': mean_loss, 'Kl loss': kl_losses})

        scores.append((train_scores, eval_scores, test_scores))

    return best_model, scores, scores[best_model_i], mean_losses


@torch.no_grad()
def get_mask(module, prune_percentage, distribution=-1, threshold=None):

    assert isinstance(module, EnsembleMaskedWrapper)

    module.set_distribution(distribution)
    _mask = module.mask

    if threshold is None:
        threshold = torch.quantile(_mask, q=prune_percentage)
    mask = torch.ge(_mask, threshold).float()

    if mask.sum() == 0:
        max = torch.argmax(_mask)
        mask = torch.zeros_like(mask)
        mask[0, max] = 1.0

    return mask


@torch.no_grad()
def register_masks(model, prune_percentage, distribution=-1, global_pruning=False):
    masks = {}

    threshold = None
    if global_pruning:
        _masks = []
        for name, module in model.named_modules():
            if isinstance(module, EnsembleMaskedWrapper):
                module.set_distribution(distribution)
                _masks.append(module.mask.detach().abs().view(-1).cpu().numpy())

        _masks = np.concatenate(_masks)
        threshold = np.quantile(_masks, q=prune_percentage)

    for name, module in model.named_modules():
        if isinstance(module, EnsembleMaskedWrapper):
            mask = get_mask(module, prune_percentage, distribution, threshold=threshold).squeeze()
            module.layer.register_buffer('mask', mask)
            masks[name] = mask

    return masks


def extract_distribution_subnetwork(model, prune_percentage, distribution, re_init=True, global_pruning=False):
    m = deepcopy(model)
    masks = register_masks(m, prune_percentage, distribution, global_pruning=global_pruning)
    masked_to_layer(m)
    extract_structured(m, re_init=re_init)

    return m, masks


def layer_to_masked(module, masks_params=None, ensemble=1, batch_ensemble=False, single_distribution=False):
    where = 'output'

    if batch_ensemble:
        wrapper = BatchEnsembleMaskedWrapper
    else:
        wrapper = EnsembleMaskedWrapper

    def apply_mask_sequential(s, skip_last):
        for i, l in enumerate(s):
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                if skip_last and i == len(s) - 1:
                    continue
                s[i] = wrapper(l, where=where, masks_params=masks_params,
                               ensemble=ensemble, single_distribution=single_distribution)

    spl = True  # if structured else False
    if isinstance(module, nn.Sequential):
        apply_mask_sequential(module, skip_last=spl)
    elif isinstance(module, VGG):
        apply_mask_sequential(module.features, skip_last=False)
        apply_mask_sequential(module.classifier, skip_last=spl)
    else:
        assert False


def masked_to_layer(model):
    def remove_masked_layer(s):
        for i, l in enumerate(s):
            if isinstance(l, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
                s[i] = l.layer

    if isinstance(model, nn.Sequential):
        remove_masked_layer(model)
    elif isinstance(model, VGG):
        remove_masked_layer(model.features)
        remove_masked_layer(model.classifier)
    else:
        assert False

    return model


def extract_structured(model, re_init=False):
    def create_layer(layer, new_w):
        if isinstance(layer, nn.Linear):
            o, i = new_w.shape
            nl = nn.Linear(in_features=i, out_features=o, bias=layer.bias is not None).to(new_w.device)
        elif isinstance(layer, nn.Conv2d):
            o, i = new_w.shape[:2]
            nl = nn.Conv2d(in_channels=i, out_channels=o, bias=layer.bias is not None,
                           kernel_size=layer.kernel_size, stride=layer.stride, padding_mode=layer.padding_mode,
                           padding=layer.padding, dilation=layer.dilation, groups=layer.groups).to(new_w.device)
        else:
            assert False

        if not re_init:
            nl.weight.data = new_w.data

        return nl

    def extract_structured_from_sequential(module, initial_mask=None):
        if not isinstance(module, nn.Sequential):
            return initial_mask

        last_mask_index = initial_mask
        for i, m in enumerate(module):
            if hasattr(m, 'weight'):
                weight = m.weight.data
                if last_mask_index is not None:
                    weight = torch.index_select(weight, 1, last_mask_index)
                    last_mask_index = None

                if hasattr(m, 'mask'):
                    mask_index = m.mask.nonzero(as_tuple=True)[0]
                    weight = torch.index_select(weight, 0, mask_index)
                    last_mask_index = mask_index

                module[i] = create_layer(m, weight)

        return last_mask_index

    if isinstance(model, nn.Sequential):
        extract_structured_from_sequential(model)
    elif isinstance(model, VGG):
        mask = extract_structured_from_sequential(model.features)
        indexes = torch.arange(0, 512*7*7, device=mask.device, dtype=torch.long)
        indexes = indexes.view((512, 7, 7))
        indexes = torch.index_select(indexes, 0, mask)
        mask = indexes.view(-1)
        extract_structured_from_sequential(model.classifier, initial_mask=mask)
    else:
        assert False

    return model


# def shrink_process():
#     add_wrappers_to_model(m, masks_params=self.supermask_parameters, ensemble=1)
#     mask_training(epochs=self.mask_epochs, model=m, device=self.device,
#                   dataset=train_dataset, ensemble=1)
#
#     grads = defaultdict(list)
#     for i, (x, y) in enumerate(train_dataset):
#         x, y = x.to(self.device), y.to(self.device)
#         pred = m(x)
#         loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')
#
#         m.zero_grad()
#         loss.backward(retain_graph=True)
#
#         for name, module in m.named_modules():
#             if isinstance(module, EnsembleMaskedWrapper):
#                 grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
#                 assert len(grad) == 1
#                 grad = grad[0]
#                 grads[name].append(torch.abs(grad).cpu())
#     m.zero_grad()
#     remove_wrappers_from_model(m)
#
#     f = lambda x: torch.mean(x, 0)
#     if self.grad_reduce == 'median':
#         f = lambda x: torch.median(x, 0)
#     elif self.grad_reduce == 'max':
#         f = lambda x: torch.max(x, 0)
#
#     ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in grads.items()}
#
#     if self.global_pruning:
#         stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in ens_grads.items()])
#         grads_sum = np.sum(stacked_grads)
#         stacked_grads = stacked_grads / grads_sum
#
#         threshold = np.quantile(stacked_grads, q=self.prune_percentage)
#
#         masks = {name: torch.ge(gs / grads_sum, threshold).float().to(self.device)
#                  for name, gs in ens_grads.items()}
#     else:
#         masks = {name: torch.ge(gs, torch.quantile(gs, self.prune_percentage)).float()
#                  for name, gs in ens_grads.items()}
#
#     for name, mask in masks.items():
#         mask = mask.squeeze()
#         if mask.sum() == 0:
#             max = torch.argmax(ens_grads[name])
#             mask = torch.zeros_like(mask)
#             mask[max] = 1.0
#         masks[name] = mask
#
#     remove_wrappers_from_model(m)
#     p1 = calculate_trainable_parameters(m)
#     m = extract_inner_model(m, masks, re_init=self.re_init)
#     p2 = calculate_trainable_parameters(m)
#     print(mi, p1, p2)
