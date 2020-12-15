from copy import deepcopy

import numpy as np
import torch
from torch import nn as nn
from torchvision.models import VGG
from tqdm import tqdm

from .layers import EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper
from .trainable_masks import MMD


def mask_training(model, epochs, dataset, ensemble, device='cpu', parameters=None):
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

            if ensemble > 1:
                for name, module in model.named_modules():
                    if isinstance(module, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
                        distr = module.distributions
                        _mmd = torch.tensor(0.0, device=device)
                        for i, d1 in enumerate(distr):
                            for j in range(i + 1, len(distr)):
                                _mmd += MMD(d1, distr[j])

                        kl += _mmd

                kl = 1 / (kl + 1e-6)
                kl *= 1e-2

            losses.append((loss.item(), kl.item()))

            loss += kl

            optim.zero_grad()
            loss.backward()
            optim.step()

        bar.set_postfix({'losses': np.mean(losses, 0)})


@torch.no_grad()
def get_mask(module, prune_percentage, distribution=-1, threshold=None):

    assert isinstance(module, EnsembleMaskedWrapper)

    module.set_distribution(distribution)
    if threshold is None:
        threshold = torch.quantile(module.mask, q=prune_percentage)
    mask = torch.ge(module.mask, threshold).float()
    if mask.sum() == 0:
        mask = torch.ge(module.mask, threshold / 2).float()

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
    _ = register_masks(m, prune_percentage, distribution, global_pruning=global_pruning)
    masked_to_layer(m)
    extract_structured(m, re_init=re_init)

    return m


def layer_to_masked(module, masks_params=None, ensemble=1, batch_ensamble=False):
    where = 'output'

    if batch_ensamble:
        wrapper = BatchEnsembleMaskedWrapper
    else:
        wrapper = EnsembleMaskedWrapper

    def apply_mask_sequential(s, skip_last):
        for i, l in enumerate(s):
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                if skip_last and i == len(s) - 1:
                    continue
                s[i] = wrapper(l, where=where, masks_params=masks_params, ensemble=ensemble)

    spl = True  # if structured else False
    if isinstance(module, nn.Sequential):
        apply_mask_sequential(module, skip_last=spl)
    elif isinstance(module, VGG):
        apply_mask_sequential(module.features, skip_last=False)
        apply_mask_sequential(module.classifier, skip_last=spl)
    else:
        assert False


def masked_to_layer(module):
    def remove_masked_layer(s):
        for i, l in enumerate(s):
            if isinstance(l, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
                s[i] = l.layer

    if isinstance(module, nn.Sequential):
        remove_masked_layer(module)
    elif isinstance(module, VGG):
        remove_masked_layer(module.features)
        remove_masked_layer(module.classifier)
    else:
        assert False


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
    optim = optimizer(model.parameters())

    train_scheduler = scheduler(optim)

    model.train()

    for e in tqdm(range(epochs)):
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.cross_entropy(pred, y, reduction='none')
            losses.extend(loss.tolist())
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        if eval_loader is not None:
            eval_scores = eval_model(model, eval_loader, device=device, topk=[1, 5])
        else:
            eval_scores = 0

        losses = sum(losses) / len(losses)