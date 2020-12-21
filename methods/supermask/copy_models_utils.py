from copy import deepcopy

import torch
from torch import nn
from torchvision.models import VGG

from methods.supermask.layers import EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper


def copy_sequential(s):
    return deepcopy(s)


def add_wrappers_to_model(module, masks_params=None, ensemble=1, batch_ensemble=False, single_distribution=False):
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


def remove_wrappers_from_model(model):
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


def extract_inner_model(model, masks, re_init=False):
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
            return module, initial_mask

        seq = []

        last_mask_index = initial_mask
        for name, m in module.named_modules():
            if isinstance(m, nn.Sequential):
                continue
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                weight = m.weight.data

                if last_mask_index is not None:
                    weight = torch.index_select(weight, 1, last_mask_index)
                    last_mask_index = None

                if name in masks:
                    mask_index = masks[name].nonzero(as_tuple=True)[0].to(m.weight.device)
                    weight = torch.index_select(weight, 0, mask_index)
                    last_mask_index = mask_index

                seq.append(create_layer(m, weight))
            else:
                seq.append(deepcopy(m))

        seq = torch.nn.Sequential(*seq)

        return seq, last_mask_index

    if isinstance(model, nn.Sequential):
        new_model, _ = extract_structured_from_sequential(model)
    elif isinstance(model, VGG):
        new_model = deepcopy(model)

        features, last_mask = extract_structured_from_sequential(new_model.features)

        indexes = torch.arange(0, 512*7*7, device=last_mask.device, dtype=torch.long)
        indexes = indexes.view((512, 7, 7))
        indexes = torch.index_select(indexes, 0, last_mask)
        last_mask = indexes.view(-1)

        classifier, _ = extract_structured_from_sequential(new_model.classifier, initial_mask=last_mask)

        new_model.features = features
        new_model.classifier = classifier

    else:
        assert False

    return new_model