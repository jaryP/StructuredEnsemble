import math
from itertools import chain

from torch import nn


def LeNet(input_size=None, output=None):
    if input_size is None:
        input_size = input_size
    if output is None:
        output = 10
    lenet = nn.Sequential(
        nn.Conv2d(in_channels=input_size, out_channels=6, kernel_size=5, stride=1, bias=False),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, bias=False),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(in_features=120, out_features=84, bias=False),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=10, bias=False),
    )

    return lenet


def LeNet_300_100(input_size=None, output=None):
    if input_size is None:
        input_size = input_size
    if output is None:
        output = 10

    lenet = nn.Sequential(
        nn.Linear(input_size, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, output)
    )

    return lenet


# '''
# Modified from https://github.com/pytorch/vision.git
# '''
# import math
#
# import torch.nn as nn
# import torch.nn.init as init

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


# class VGG(nn.Module):
#     '''
#     VGG model
#     '''
#     def __init__(self, features):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10),
#         )
#          # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#
# def make_layers(cfg, output, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#
#     layers.append(nn.Flatten())
#     layers.extend([
#         nn.Dropout(),
#         nn.Linear(512, 512),
#         nn.ReLU(True),
#         nn.Dropout(),
#         nn.Linear(512, 512),
#         nn.ReLU(True),
#         nn.Linear(512, output),
#     ])
#
#     layers = nn.Sequential(*layers)
#
#     for m in layers:
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             m.bias = None
#
#     return layers
#
#
# def vgg(name, output):
#     cfg = {
#         'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#         'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#         'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#         'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
#               512, 512, 512, 512, 'M'],
#     }
#
#     if name == 'vgg11':
#         return make_layers(cfg['A'], output)

# def vgg11():
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers(cfg['A']))
#
#
# def vgg11_bn():
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cfg['A'], batch_norm=True))
#
#
# def vgg13():
#     """VGG 13-layer model (configuration "B")"""
#     return VGG(make_layers(cfg['B']))
#
#
# def vgg13_bn():
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg['B'], batch_norm=True))
#
#
# def vgg16():
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg['D']))
#
#
# def vgg16_bn():
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(make_layers(cfg['D'], batch_norm=True))
#
#
# def vgg19():
#     """VGG 19-layer model (configuration "E")"""
#     return VGG(make_layers(cfg['E']))
#
#
# def vgg19_bn():
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG(make_layers(cfg['E'], batch_norm=True))
