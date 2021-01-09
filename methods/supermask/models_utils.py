from copy import deepcopy
import torch
from torch import nn
from torch.nn import BatchNorm2d
from torchvision.models import VGG
import numpy as np
# from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn.functional as F

from methods.supermask.layers import EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper
from methods.supermask.resnet_utils import ResNetBlockWrapper
from models import ResNet, BasicBlock


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
            elif isinstance(l, BasicBlock):
                s[i] = ResNetBlockWrapper(l, masks_params=masks_params,
                                          ensemble=ensemble, batch_ensemble=batch_ensemble)

    spl = True  # if structured else False
    if isinstance(module, nn.Sequential):
        apply_mask_sequential(module, skip_last=True)
    elif isinstance(module, VGG):
        apply_mask_sequential(module.features, skip_last=False)
        apply_mask_sequential(module.classifier, skip_last=True)
    elif isinstance(module, ResNet):
        module.conv1 = wrapper(module.conv1, masks_params=masks_params, where='output',
                               ensemble=ensemble, batch_ensemble=batch_ensemble)
        # module.fc = wrapper(module.fc, masks_params=masks_params, where='output',
        #                     ensemble=ensemble, batch_ensemble=batch_ensemble)
        for i in range(1, 4):
            apply_mask_sequential(getattr(module, 'layer{}'.format(i)), skip_last=True)
    else:
        assert False


def remove_wrappers_from_model(model):
    def remove_masked_layer(s):
        for i, l in enumerate(s):
            if isinstance(l, (EnsembleMaskedWrapper, BatchEnsembleMaskedWrapper)):
                s[i] = l.layer
            if isinstance(l, ResNetBlockWrapper):
                s[i] = l.block
                if isinstance(l.block.shortcut, nn.Sequential):
                    if len(l.block.shortcut) > 0:
                     # if l.block.downsample is not None:
                        s[i].shortcut[0] = l.block.shortcut[0].layer

    if isinstance(model, nn.Sequential):
        remove_masked_layer(model)
    elif isinstance(model, VGG):
        remove_masked_layer(model.features)
        remove_masked_layer(model.classifier)
    elif isinstance(model, ResNet):
        model.conv1 = model.conv1.layer
        # model.fc = model.fc.layer
        for i in range(1, 4):
            remove_masked_layer(getattr(model, 'layer{}'.format(i)))
    else:
        assert False

    return model


def extract_inner_model(model, masks, re_init=False):
    def extract_weights(weight, mask):
        mask_index = mask.nonzero(as_tuple=True)[0].to(weight.device)
        weight = torch.index_select(weight, 0, mask_index)
        return weight, mask_index

    def create_layer(layer, new_w):
        if isinstance(layer, nn.Linear):
            o, i = new_w.shape
            nl = nn.Linear(in_features=i, out_features=o, bias=layer.bias is not None).to(new_w.device)
        elif isinstance(layer, nn.Conv2d):
            o, i = new_w.shape[:2]
            nl = nn.Conv2d(in_channels=i, out_channels=o, bias=layer.bias is not None,
                           kernel_size=layer.kernel_size, stride=layer.stride, padding_mode=layer.padding_mode,
                           padding=layer.padding, dilation=layer.dilation, groups=layer.groups).to(new_w.device)
        elif isinstance(layer, BatchNorm2d):
            w, v = new_w
            inp = w.size(0)
            nl = nn.BatchNorm2d(inp)
        else:
            assert False

        if not re_init and not isinstance(layer, BatchNorm2d):
            nl.weight.data = new_w.data

        return nl

    def create_new_block(block, block_mask, input_indexes):

        hidden_dim, input_dim, _, _ = block.conv1.weight.shape
        output_dim, _, _, _ = block.conv2.weight.shape

        weight = block.conv1.weight
        if input_indexes is not None:
            weight = torch.index_select(weight, 1, input_indexes)
        conv1_weight, conv1_indexes = extract_weights(weight, block_mask['conv1'])

        conv2_weight = block.conv2.weight
        conv2_weight = torch.index_select(conv2_weight, 1, conv1_indexes)
        if input_indexes is not None:
            conv2_weight = torch.index_select(conv2_weight, 0, input_indexes)

        new_hidden_dim, new_input_dim, _, _ = conv1_weight.shape
        new_output_dim, _, _, _ = conv2_weight.shape
        
        new_block = BasicBlock(in_planes=new_input_dim, planes=new_output_dim, hidden_planes=new_hidden_dim,
                               stride=block.stride)
        if not re_init:
            new_block.conv1.weight.data = conv1_weight.data
            new_block.conv2.weight.data = conv2_weight.data

        return new_block, input_indexes

    def extract_structured_from_block(block, block_masks, input_indexes):

        # extract the weights based on the input dimension (using the associated mask)
        weight = block.conv1.weight
        weight = torch.index_select(weight, 1, input_indexes)
        weight, conv1_indexes = extract_weights(weight, block_masks['conv1'])
        block.conv1 = create_layer(block.conv1, weight)

        # extract the batch norm based on the input dimension (using the associated mask)
        m, v = block.bn1.running_mean, block.bn1.running_var
        m = torch.index_select(m, 0, conv1_indexes)
        v = torch.index_select(v, 0, conv1_indexes)
        block.bn1 = create_layer(block.bn1, (m, v))

        conv2_weight = block.conv2.weight
        conv2_weight = torch.index_select(conv2_weight, 1, conv1_indexes)

        # if downsample than we need to calcualte the new indexes to extract the weights in the following layers
        if 'block.downsample.0' in block_masks:
            weight = block.downsample[0].weight
            weight = torch.index_select(weight, 1, input_indexes)
            weight, input_indexes = extract_weights(weight, block_masks['block.downsample.0'])
            block.downsample[0] = create_layer(block.downsample[0], weight)

            m, v = block.downsample[1].running_mean, block.downsample[1].running_var
            m = torch.index_select(m, 0, input_indexes)
            v = torch.index_select(v, 0, input_indexes)

            block.downsample[1] = create_layer(block.downsample[1], (m, v))
        else:
            # print(block.shortcut, isinstance(block.shortcut, LambdaLayer))
            if isinstance(block.shortcut, LambdaLayer):
                planes = block.conv1.weight.shape[0]
                print(block.conv1.weight.shape)
                block.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                                                      "constant", 0))

        # extract the weights in the second layer based on the input dimension (using the associated mask)
        # weight, conv1_indexes = extract_weights(conv2_weight, masks['conv1'])
        # weight_conv2 = torch.index_select(block.conv2.weight, 1, conv1_indexes)
        conv2_weight = torch.index_select(conv2_weight, 0, input_indexes)
        block.conv2 = create_layer(block.conv2, conv2_weight)

        m, v = block.bn2.running_mean, block.bn2.running_var
        m = torch.index_select(m, 0, input_indexes)
        v = torch.index_select(v, 0, input_indexes)

        block.bn2 = create_layer(block.bn2, (m, v))

        return block, input_indexes

    def extract_structured_from_sequential(module, initial_mask=None, prefix=''):
        if not isinstance(module, nn.Sequential):
            return module, initial_mask

        seq = []

        last_mask_index = initial_mask
        for name, m in module.named_modules():
            name = prefix+name
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

        prefix = 'features.'
        features, last_mask = extract_structured_from_sequential(new_model.features, prefix='features.')

        indexes = torch.arange(0, 512 * 7 * 7, device=last_mask.device, dtype=torch.long)
        indexes = indexes.view((512, 7, 7))
        indexes = torch.index_select(indexes, 0, last_mask)
        last_mask = indexes.view(-1)

        classifier, _ = extract_structured_from_sequential(new_model.classifier, initial_mask=last_mask,
                                                           prefix='classifier.')

        new_model.features = features
        new_model.classifier = classifier

    elif isinstance(model, ResNet):
        new_model = deepcopy(model)
        mask_indexes = None
        # first_mask = masks['conv1']
        # weights, mask_indexes = extract_weights(new_model.conv1.weight.data, first_mask)
        # new_model.conv1 = create_layer(new_model.conv1, weights)
        #
        # m, v = new_model.bn1.running_mean, new_model.bn1.running_var
        # m = torch.index_select(m, 0, mask_indexes)
        # v = torch.index_select(v, 0, mask_indexes)
        # new_model.bn1 = create_layer(new_model.bn1, (m, v))

        for i in range(1, 4):
            l = getattr(new_model, 'layer{}'.format(i))
            for si, s in enumerate(l):
                block_name = 'layer{}.{}'.format(i, si)
                block_masks = {name[len(block_name)+1:]: mask for name, mask in masks.items() if block_name in name}
                # block, mask_indexes = extract_structured_from_block(deepcopy(s), input_indexes=mask_indexes,
                #                                                     block_masks=block_masks)
                new_block, mask_indexes = create_new_block(s, block_masks, mask_indexes)
                l[si] = new_block

        if mask_indexes is not None:
            fc = new_model.fc
            weight = torch.index_select(fc.weight, 1, mask_indexes)
            new_model.fc = create_layer(fc, weight)
    else:
        assert False

    return new_model


def get_masks_from_gradients(gradients, prune_percentage, global_pruning, device='cpu'):
    if global_pruning:
        stacked_grads = np.concatenate([gs.view(-1).numpy() for name, gs in gradients.items()])
        grads_sum = np.sum(stacked_grads)
        stacked_grads = stacked_grads / grads_sum

        threshold = np.quantile(stacked_grads, q=prune_percentage)

        masks = {name: torch.ge(gs / grads_sum, threshold).float().to(device)
                 for name, gs in gradients.items()}
    else:
        masks = {name: torch.ge(gs, torch.quantile(gs, prune_percentage)).float()
                 for name, gs in gradients.items()}

    for name, mask in masks.items():
        mask = mask.squeeze()
        if mask.sum() == 0:
            max = torch.argmax(gradients[name])
            mask = torch.zeros_like(mask)
            mask[max] = 1.0
        masks[name] = mask

    return masks


if __name__ == '__main__':
    import torch

    import torchvision.models as models
    from collections import defaultdict
    from utils import calculate_trainable_parameters
    from models import resnet20, LambdaLayer

    x = torch.rand((12, 3, 32, 32))
    y = torch.randint(9, size=(12,))

    resnet18 = resnet20(num_classes=10)
    outputs = resnet18(x)

    add_wrappers_to_model(resnet18, ensemble=2,
                          masks_params={'name': 'weights', 'initialization': {'name': 'constant', 'c': 1}},
                          batch_ensemble=True)

    p = calculate_trainable_parameters(resnet18)
    print(p)

    x = torch.cat([x for _ in range(2)], dim=0)

    outputs = resnet18(x)
    outputs = outputs.view([2, 12, -1])
    pred = torch.mean(outputs, 0)

    loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')

    resnet18.zero_grad()
    loss.backward(retain_graph=True)

    grads = defaultdict(lambda: defaultdict(list))

    for name, module in resnet18.named_modules():
        if isinstance(module, BatchEnsembleMaskedWrapper):
            grad = torch.autograd.grad(loss, module.last_mask, retain_graph=True)
            for i, g in enumerate(grad):
                grads[i][name].append(torch.abs(g).cpu())
        # elif isinstance(module, ResNetBlockWrapper):

    # print(grads[0].keys())

    remove_wrappers_from_model(resnet18)

    for mi, ens_grads in grads.items():
        f = lambda x: torch.mean(x, 0)

        ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in ens_grads.items()}

        masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=0.8,
                                         global_pruning=True)

        model = extract_inner_model(resnet18, masks, False)
        print(model)

        p = calculate_trainable_parameters(model)
        print(mi, p)

        model(x)

    # print(resnet18)
    # x = torch.rand((12, 3, 224, 224))
    # r = resnet18(x)
    # remove_wrappers_from_model(resnet18)
    # r = resnet18(x)
    # print(r.shape)
    # print(resnet18)


