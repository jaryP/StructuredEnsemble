from copy import deepcopy
from typing import Union

import numpy as np
import torch
from torch import nn as nn
from torchvision.models import VGG

from .trainable_masks import TrainableBeta, TrainableWeights, TrainableLaplace


class EnsembleMaskedWrapper(nn.Module):
    def __init__(self, layer: Union[nn.Linear, nn.Conv2d], where: str, masks_params: dict, ensemble: int = 1,
                 t: int = 1):

        super().__init__()

        self._use_mask = True
        self._eval_mask = None

        self.where = where.lower()
        self.masks = []
        self.steps = 0

        self.layer = layer

        mask_dim = layer.weight.shape
        self.is_conv = isinstance(layer, nn.Conv2d)

        self._current_distribution = -1

        where = where.lower()
        if not where == 'weights':
            if where == 'output':
                mask_dim = (1, mask_dim[0])
            else:
                assert False, 'The following types are allowed: output, input and weights. {} given'.format(where)

            if self.is_conv:
                mask_dim = mask_dim + (1, 1)

        self._distributions = nn.ModuleList()

        for i in range(ensemble):
            if masks_params['name'] == 'beta':
                distribution = TrainableBeta(mask_dim, t=t, initialization=masks_params['initialization'])
            elif masks_params['name'] == 'laplace':
                distribution = TrainableLaplace(mask_dim, t=t, initialization=masks_params['initialization'])
            elif masks_params['name'] == 'weights':
                distribution = TrainableWeights(mask_dim, initialization=masks_params['initialization'])
            else:
                assert False

            distribution.to(layer.weight.device)
            self._distributions.append(distribution)

    @property
    def distributions(self):
        return self._distributions

    def set_distribution(self, v):
        assert v <= len(self.distributions) or v == 'all'
        if v == 'all':
            v = -1
        self._current_distribution = v

    @property
    def apply_mask(self):
        return self._use_mask

    @apply_mask.setter
    def apply_mask(self, v: bool):
        self._use_mask = v

    def posterior(self):
        return self.distribution.posterior

    @property
    def mask(self):

        if not self.apply_mask:
            # if len(self.masks) > 0:
            #     return self.masks[-1]
            return 1

        # if self._eval_mask is not None:
        #     return self._eval_mask

        if self._current_distribution < 0:
            masks = [d(reduce=True) for d in self._distributions]
            m = sum(masks)
        else:
            m = self._distributions[self._current_distribution](reduce=True)

        return m

    def eval(self):
        self._eval_mask = self.mask
        return self.train(False)

    def train(self, mode=True):
        self._eval_mask = None
        return super().train(mode)

    # TODO: implemetare MMD
    def calculate_kl(self, prior: torch.distributions.Distribution):
        if not self.apply_mask:
            return 0

        kl = self.distribution.calculate_divergence(prior).sum()
        # kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def forward(self, x):
        if self.where == 'input':
            x = self.mask * x

        if self.where == 'weights':
            w = self.mask * self.layer.weight
        else:
            w = self.layer.weight

        if self.is_conv:
            o = nn.functional.conv2d(x, w, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding,
                                     dilation=self.layer.dilation, groups=self.layer.groups)
        else:
            o = nn.functional.linear(x, w, self.layer.bias)

        if self.where == 'output':
            o = o * self.mask

        return o

    def __repr__(self):
        return 'Supermask {} layer with distribution {}. ' \
               'Original layer: {} '.format('structured' if self.where != 'weights' else 'unstructured',
                                            self.distribution, self.layer.__repr__())


@torch.no_grad()
def get_mask(module, prune_percentage, distribution=-1, threshold=None):

    assert isinstance(module, EnsembleMaskedWrapper)

    module.set_distribution(distribution)
    if threshold is None:
        threshold = torch.quantile(module.mask, q=prune_percentage)
    mask = torch.ge(module.mask, threshold).float()

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


def layer_to_masked(module, masks_params=None, ensemble=1):
    where = 'output'

    def apply_mask_sequential(s, skip_last):
        for i, l in enumerate(s):
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                if skip_last and i == len(s) - 1:
                    continue
                s[i] = EnsembleMaskedWrapper(l, where=where, masks_params=masks_params, ensemble=ensemble)

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
            if isinstance(l, EnsembleMaskedWrapper):
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