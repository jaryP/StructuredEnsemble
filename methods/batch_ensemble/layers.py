import torch
from torch import nn
from torchvision.models import VGG


def layer_to_masked(module, ensemble=1):
    def apply_mask_sequential(s):
        for i, l in enumerate(s):
            if isinstance(l, nn.Linear):
                s[i] = BELinear(l, ensemble=ensemble)
            elif isinstance(l, nn.Conv2d):
                s[i] = BEConv2D(l, ensemble=ensemble)

    if isinstance(module, nn.Sequential):
        apply_mask_sequential(module)
    elif isinstance(module, VGG):
        apply_mask_sequential(module.features)
        apply_mask_sequential(module.classifier)
    else:
        assert False


class BELinear(torch.nn.Module):
    def __init__(self, layer: nn.Linear, ensemble: int):
        super().__init__()

        self.ensemble = ensemble
        alpha = torch.ones((ensemble, layer.in_features))
        self.alpha = nn.Parameter(alpha, requires_grad=True)

        gamma = torch.ones((ensemble, layer.out_features))
        self.gamma = nn.Parameter(gamma, requires_grad=True)

        self.input_dim = layer.in_features
        self.out_dim = layer.out_features
        self.layer = layer

    def forward(self, x):
        batch_size = x.size(0)
        rest = batch_size % self.ensemble
        m = batch_size // self.ensemble

        alpha = self.alpha.repeat(1, m).view(-1,  self.input_dim)
        gamma = self.gamma.repeat(1, m).view(-1,  self.out_dim)

        if rest > 0:
            alpha = torch.cat([alpha, alpha[:rest]], dim=0)
            gamma = torch.cat([gamma, gamma[:rest]], dim=0)

        x = x * alpha
        output = self.layer(x) * gamma
        return output

    def __repr__(self):
        return 'Batch ensemble. Original layer: {} '.format(self.layer.__repr__())


class BEConv2D(torch.nn.Module):
    def __init__(self, layer: nn.Conv2d, ensemble: int):
        super().__init__()

        self.ensemble = ensemble
        alpha = torch.ones((ensemble, layer.in_channels))
        self.alpha = nn.Parameter(alpha, requires_grad=True)

        gamma = torch.ones((ensemble, layer.out_channels))
        self.gamma = nn.Parameter(gamma, requires_grad=True)

        self.input_dim = layer.in_channels
        self.out_dim = layer.out_channels

        self.layer = layer

    def forward(self, x):
        batch_size = x.size(0)
        m = batch_size // self.ensemble
        rest = batch_size % self.ensemble

        alpha = self.alpha.repeat(1, m).view(-1, self.input_dim)
        gamma = self.gamma.repeat(1, m).view(-1, self.out_dim)

        if rest > 0:
            alpha = torch.cat([alpha, alpha[:rest]], dim=0)
            gamma = torch.cat([gamma, gamma[:rest]], dim=0)

        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        x = x * alpha
        output = self.layer(x) * gamma

        return output

    def __repr__(self):
        return 'Batch ensemble. Original layer: {} '.format(self.layer.__repr__())
