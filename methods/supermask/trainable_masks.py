from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal, Uniform, Laplace, Exponential, Gamma


def gaussian_kernel(a, b, gamma=None):
    a = a.squeeze()
    b = b.squeeze()

    assert len(a.shape) == len(b.shape) == 1
    if gamma is None:
        gamma = a.shape[0]
    diff = a.unsqueeze(1) - b.unsqueeze(0)

    numerator = diff.pow(2).mean(1)/gamma
    return torch.exp(-numerator)


def MMD(a, b):

    if isinstance(a, TrainableMask):
        a = a()
    if isinstance(b, TrainableMask):
        b = b()

    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()


def KL(posterior, prior):
   return torch.distributions.kl.kl_divergence(posterior, prior).sum()


def get_distribution(name, **kwargs):
    name = name.lower()
    if name == 'normal':
        distribution = Normal(kwargs['mu'], kwargs['std'])
    elif name == 'uniform':
        distribution = Uniform(kwargs['low'], kwargs['high'])
    elif name == 'beta':
        distribution = Beta(kwargs['a'], kwargs['b'])
    elif name == 'laplace':
        distribution = Laplace(kwargs['a'], kwargs['b'])
    else:
        assert False

    return distribution


def get_trainable_mask(size, initial_distribution):
    if initial_distribution['name'].lower() == 'constant':
        t = torch.empty(size).fill_(initial_distribution['c'])
    else:
        t = get_distribution(**initial_distribution).sample(size)

    if initial_distribution.get('trainable', True):
        return nn.Parameter(t, requires_grad=True)
    return t


class TrainableMask(nn.Module, ABC):
    def __init__(self, t=1):
        super().__init__()
        self.t = t

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calculate_divergence(self, prior: torch.distributions.Distribution):
        raise NotImplementedError


class TrainableBeta(TrainableMask):
    def __init__(self, dimensions, initialization, eps=1e-6, t=1):
        super().__init__(t)
        self.a = get_trainable_mask(dimensions, initialization['a'])
        self.b = get_trainable_mask(dimensions, initialization['b'])
        self.eps = eps

    @property
    def posterior(self) -> torch.distributions.Distribution:
        a, b = torch.relu(self.a) + self.eps, torch.relu(self.b) + self.eps
        return Beta(a, b)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return '{}(a={}, b={})'.format(self.__class__.__name__, tuple(self.a.shape), tuple(self.b.shape))


class TrainableGamma(TrainableMask):
    def __init__(self, dimensions, initialization, eps=1e-6, t=1):
        super().__init__(t)
        self.a = get_trainable_mask(dimensions, initialization['a'])
        self.b = get_trainable_mask(dimensions, initialization['b'])
        self.eps = eps

    @property
    def posterior(self) -> torch.distributions.Distribution:
        a, b = torch.relu(self.a) + self.eps, torch.relu(self.b) + self.eps
        return Gamma(a, b)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return '{}(a={}, b={})'.format(self.__class__.__name__, tuple(self.a.shape), tuple(self.b.shape))


class TrainableLaplace(TrainableMask):
    def __init__(self, dimensions, initialization, t=1):
        super().__init__(t)
        self.mu = get_trainable_mask(dimensions, initialization['mu'])
        self.b = get_trainable_mask(dimensions, initialization['b'])

    @property
    def posterior(self) -> torch.distributions.Distribution:
        return Laplace(self.mu, self.b)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return '{}(a={}, b={})'.format(self.__class__.__name__, tuple(self.mu.shape), tuple(self.b.shape))


class TrainableNormal(TrainableMask):
    def __init__(self, dimensions, initialization, t=1):
        super().__init__(t)
        self.mu = get_trainable_mask(dimensions, initialization['mu'])
        self.std = get_trainable_mask(dimensions, initialization['std'])

    @property
    def posterior(self) -> torch.distributions.Distribution:
        return Normal(self.mu, self.std)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return '{}(a={}, b={})'.format(self.__class__.__name__, tuple(self.mu.shape), tuple(self.b.shape))


class TrainableWeights(TrainableMask):
    def __init__(self, dimensions, initialization):
        super().__init__()
        self.w = get_trainable_mask(dimensions, initialization)

    def __call__(self, *args, **kwargs):
        return self.w

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        # kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return 0

    def __repr__(self):
        return '{}(dim={})'.format(self.__class__.__name__, tuple(self.w.shape))


class TrainableExponential(TrainableMask):
    def __init__(self, dimensions, initialization, eps=1e-12, max=None, t=1):
        super().__init__(t)
        self.l = get_trainable_mask(dimensions, initialization)
        self.eps = eps
        self.max = max

    @property
    def posterior(self) -> torch.distributions.Distribution:
        l = torch.clamp(self.l, min=self.eps, max=self.max)
        return Exponential(l)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return '{}(l={})'.format(self.__class__.__name__, tuple(self.l.shape))
