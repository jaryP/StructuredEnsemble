from typing import Union

import torch
from torch import nn as nn

from methods.supermask.trainable_masks import TrainableBeta, TrainableLaplace, TrainableWeights


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

        self.distributions = nn.ModuleList()

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
            self.distributions.append(distribution)

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
            return 1

        # if self._eval_mask is not None:
        #     return self._eval_mask

        if self._current_distribution < 0:
            masks = [d(reduce=True) for d in self.distributions]
            m = torch.mean(torch.stack(masks), 0)
        else:
            m = self.distributions[self._current_distribution](reduce=True)

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
                                            self.distributions, self.layer.__repr__())


class BatchEnsembleMaskedWrapper(nn.Module):
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
                mask_dim = (mask_dim[0], )
                # mask_dim = (1, mask_dim[0])
            else:
                assert False, 'The following types are allowed: output, input and weights. {} given'.format(where)

            # if self.is_conv:
            #     mask_dim = mask_dim + (1, 1)

        self.distributions = nn.ModuleList()

        #TODO: fare in modo che t = n_ensambles

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
            self.distributions.append(distribution)

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

    # @property
    # def mask(self):
    #
    #     if not self.apply_mask:
    #         return 1
    #
    #     # if self._eval_mask is not None:
    #     #     return self._eval_mask
    #
    #     if self._current_distribution < 0:
    #         masks = [d(reduce=True) for d in self.distributions]
    #         m = torch.mean(torch.stack(masks), 0)
    #     else:
    #         m = self.distributions[self._current_distribution](reduce=True)
    #
    #     return m

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
        # if self.where == 'input':
        #     x = self.mask * x
        #
        # if self.where == 'weights':
        #     w = self.mask * self.layer.weight
        # else:

        w = self.layer.weight

        if self.is_conv:
            o = nn.functional.conv2d(x, w, self.layer.bias, stride=self.layer.stride, padding=self.layer.padding,
                                     dilation=self.layer.dilation, groups=self.layer.groups)
        else:
            o = nn.functional.linear(x, w, self.layer.bias)

        if self.where == 'output':
            batch_size = x.size(0)
            ensemble = len(self.distributions)
            m = batch_size // ensemble
            rest = batch_size % ensemble

            masks = [d(reduce=True) for d in self.distributions]
            masks = torch.stack(masks, 0)
            masks = masks.repeat(1, m).view(-1, masks.size(-1))

            if self.is_conv:
                masks = masks.unsqueeze(-1).unsqueeze(-1)

            if rest > 0:
                masks = torch.cat([masks, masks[:rest]], dim=0)

            o = o * masks

        return o

    def __repr__(self):
        return 'Supermask {} layer with distribution {}. ' \
               'Original layer: {} '.format('structured' if self.where != 'weights' else 'unstructured',
                                            self.distributions, self.layer.__repr__())