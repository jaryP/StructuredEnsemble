import os
from copy import deepcopy

import torch
from torch import nn
from torchvision.models import VGG

from methods.base import EnsembleMethod
from models import ResNet
from utils import train_model


class DropoutWrapper(nn.Module):
    def __init__(self, layer, p, inplace=False):
        super().__init__()
        self.p = p
        self.layer = layer
        self.inplace = inplace

    def forward(self, x):
        o = self.layer(x)
        o = nn.functional.dropout(o, self.p, True, self.inplace)
        return o


def wrap_dropout(model, p, inplace=False):
    def apply_mask_sequential(s, skip_last=False):
        for i, l in enumerate(s):
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                if skip_last and i == len(s) - 1:
                    continue
                s[i] = DropoutWrapper(l, p=p, inplace=inplace)

    if isinstance(model, nn.Sequential):
        apply_mask_sequential(model, skip_last=True)
    elif isinstance(model, VGG):
        apply_mask_sequential(model.features, skip_last=False)
        apply_mask_sequential(model.classifier, skip_last=True)
    elif isinstance(model, ResNet):
        model.conv1 = DropoutWrapper(model.conv1, p=p, inplace=inplace)
        for i in range(1, 4):
            apply_mask_sequential(getattr(model, 'layer{}'.format(i)), skip_last=True)
    else:
        assert False


class MCDropout(EnsembleMethod):
    def __init__(self, model, method_parameters,
                 device='cpu', **kwargs):

        super().__init__(**kwargs)

        self.method_parameters = method_parameters

        self.device = device

        self.training_t = method_parameters.get('train_sample', 1)
        self.testing_t = method_parameters.get('test_sample', 1)
        p = method_parameters.get('p', 0.2)
        inplace = method_parameters.get('inplace', False)

        self.model = deepcopy(model)
        wrap_dropout(self.model, p=p, inplace=inplace)

    def train_models(self, epochs, train_dataset, eval_dataset, test_dataset, optimizer, scheduler=None,
              regularization=None, early_stopping=None,
              **kwargs):

        optim = optimizer([param for name, param in self.model.named_parameters() if param.requires_grad])
        train_scheduler = scheduler(optim)

        best_model, scores, best_model_scores, losses = train_model(model=self.model, optimizer=optim,
                                                            epochs=epochs, train_loader=train_dataset,
                                                            scheduler=train_scheduler,
                                                            early_stopping=early_stopping,
                                                            test_loader=test_dataset, eval_loader=eval_dataset,
                                                            device=self.device, t=self.training_t)

        self.model.load_state_dict(best_model)

        return [scores]

    def predict_logits(self, x, y, reduce=True, **kwargs):
        x, y = x.to(self.device), y.to(self.device)
        outputs = [self.model(x) for _ in range(self.testing_t)]
        outputs = torch.stack(outputs, dim=1)
        # print(outputs.shape)
        if reduce:
            outputs = torch.mean(outputs, 1)
        # outputs = torch.nn.functional.softmax(outputs, -1)
        return outputs

    def load(self, path):
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
