from copy import deepcopy

from torch import nn as nn
from torch.nn import BatchNorm2d

from methods.supermask.layers import BatchEnsembleMaskedWrapper, EnsembleMaskedWrapper


class ResNetBlockWrapper(nn.Module):
    def __init__(self, block, masks_params: dict, ensemble: int = 1,
                 t: int = 1, batch_ensemble=True, ):
        super().__init__()

        if batch_ensemble:
            wrapper = BatchEnsembleMaskedWrapper
        else:
            wrapper = EnsembleMaskedWrapper

        self.block = block
        self.masks_params = masks_params

        self.conv1 = wrapper(block.conv1, where='output', masks_params=masks_params,
                             ensemble=ensemble, t=t)

        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = wrapper(block.conv2, where='output', masks_params=masks_params,
        #                      ensemble=ensemble, t=t)
        self.conv2 = self.block.conv2

        self.shortcut = block.shortcut

        if isinstance(self.shortcut, nn.Sequential):
            if len(self.shortcut) > 0:
                # pass
                # else:
                # if block.downsample is not None:
                self.shortcut[0] = wrapper(self.downsample[0], where='output', masks_params=masks_params,
                                             ensemble=ensemble)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.block.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.block.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

# class ResNetBottleneckWrapper(nn.Module):
#     def __init__(self, bottleneck):
#         super().__init__()
#         self.bottleneck = bottleneck

#
# def extract_inner_model(model, masks, re_init=False, **kwargs):
#     def extract_weights(weight, mask):
#         mask_index = mask.nonzero(as_tuple=True)[0].to(weight.device)
#         weight = torch.index_select(weight, 0, mask_index)
#         return weight, mask_index
#
#     def create_layer(layer, new_w):
#         if isinstance(layer, nn.Linear):
#             o, i = new_w.shape
#             nl = nn.Linear(in_features=i, out_features=o, bias=layer.bias is not None).to(new_w.device)
#         elif isinstance(layer, nn.Conv2d):
#             o, i = new_w.shape[:2]
#             nl = nn.Conv2d(in_channels=i, out_channels=o, bias=layer.bias is not None,
#                            kernel_size=layer.kernel_size, stride=layer.stride, padding_mode=layer.padding_mode,
#                            padding=layer.padding, dilation=layer.dilation, groups=layer.groups).to(new_w.device)
#         elif isinstance(layer, BatchNorm2d):
#             w, v = new_w
#             inp = w.size(0)
#             nl = nn.BatchNorm2d(inp)
#         else:
#             assert False
#
#         if not re_init:
#             if isinstance(layer, BatchNorm2d):
#                 nl.running_mean.data = new_w[0]
#                 nl.running_var.data = new_w[1]
#             else:
#                 nl.weight.data = new_w.data
#
#         return nl
#
#     def extract_structured_from_sequential(module, initial_mask=None):
#         if not isinstance(module, nn.Sequential):
#             return module, initial_mask
#
#         seq = []
#
#         last_mask_index = initial_mask
#         for name, m in module.named_modules():
#             if isinstance(m, nn.Sequential):
#                 continue
#             if isinstance(m, (nn.Linear, nn.Conv2d)):
#                 weight = m.weight.data
#
#                 if last_mask_index is not None:
#                     weight = torch.index_select(weight, 1, last_mask_index)
#                     last_mask_index = None
#
#                 if name in masks:
#                     weight, mask_index = extract_weights(weight, masks[name])
#                     last_mask_index = mask_index
#
#                 seq.append(create_layer(m, weight))
#             else:
#                 seq.append(deepcopy(m))
#
#         seq = torch.nn.Sequential(*seq)
#
#         return seq, last_mask_index
#
#     def extract_structured_from_block(block, block_masks, input_indexes):
#
#         # extract the weights based on the input dimension (using the associated mask)
#         weight = block.conv1.weight
#         weight = torch.index_select(weight, 1, input_indexes)
#         weight, conv1_indexes = extract_weights(weight, masks['conv1'])
#         block.conv1 = create_layer(block.conv1, weight)
#
#         # extract the batch norm based on the input dimension (using the associated mask)
#         m, v = block.bn1.running_mean, block.bn1.running_var
#         m = torch.index_select(m, 0, conv1_indexes)
#         v = torch.index_select(v, 0, conv1_indexes)
#         block.bn1 = create_layer(block.bn1, (m, v))
#
#         # extract the weights in the second layer based on the input dimension (using the associated mask)
#         weight_conv2 = torch.index_select(block.conv2.weight, 1, conv1_indexes)
#
#         # if downsample than we need to calcualte the new indexes to extract the weights in the following layers
#         if 'block.downsample.0' in block_masks:
#             weight = block.downsample[0].weight
#             weight = torch.index_select(weight, 1, input_indexes)
#             weight, input_indexes = extract_weights(weight, block_masks['block.downsample.0'])
#             block.downsample[0] = create_layer(block.downsample[0], weight)
#
#             m, v = block.downsample[1].running_mean, block.downsample[1].running_var
#             m = torch.index_select(m, 0, input_indexes)
#             v = torch.index_select(v, 0, input_indexes)
#
#             block.downsample[1] = create_layer(block.downsample[1], (m, v))
#
#         weight = torch.index_select(weight_conv2, 0, input_indexes)
#         block.conv2 = create_layer(block.conv2, weight)
#
#         m, v = block.bn2.running_mean, block.bn2.running_var
#         m = torch.index_select(m, 0, input_indexes)
#         v = torch.index_select(v, 0, input_indexes)
#
#         block.bn2 = create_layer(block.bn2, (m, v))
#
#         return block, input_indexes
#
#     if isinstance(model, models.ResNet):
#         new_model = deepcopy(model)
#
#         first_mask = masks['conv1']
#         weights, mask_indexes = extract_weights(new_model.conv1.weight.data, first_mask)
#         new_model.conv1 = create_layer(new_model.conv1, weights)
#
#         m, v = new_model.bn1.running_mean, new_model.bn1.running_var
#         m = torch.index_select(m, 0, mask_indexes)
#         v = torch.index_select(v, 0, mask_indexes)
#         new_model.bn1 = create_layer(new_model.bn1, (m, v))
#
#         for i in range(1, 5):
#             l = getattr(new_model, 'layer{}'.format(i))
#             for si, s in enumerate(l):
#                 block_name = 'layer{}.{}'.format(i, si)
#                 block_masks = {name[len(block_name)+1:]: mask for name, mask in masks.items() if block_name in name}
#                 block, mask_indexes = extract_structured_from_block(s, input_indexes=mask_indexes,
#                                                                     block_masks=block_masks)
#                 l[si] = block
#
#         fc = new_model.fc
#         weight = torch.index_select(fc.weight, 1, mask_indexes)
#         new_model.fc = create_layer(fc, weight)
#     else:
#         assert False
#
#     return new_model


if __name__ == '__main__':
    import torch
    from models import resnet20, LambdaLayer

    # import torchvision.models as models
    from collections import defaultdict
    from utils import calculate_trainable_parameters
    from methods.supermask.models_utils import remove_wrappers_from_model, get_masks_from_gradients, \
    add_wrappers_to_model, extract_inner_model

    resnet18 = resnet20(num_classes=10)

    add_wrappers_to_model(resnet18, ensemble=2,
                          masks_params={'name': 'weights', 'initialization': {'name': 'constant', 'c': 1}},
                          batch_ensemble=True)

    x = torch.rand((12, 3, 32, 32))
    y = torch.randint(9, size=(12,))

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

    print(grads[0].keys())
    print(resnet18)

    remove_wrappers_from_model(resnet18)

    # print(resnet18)

    # print(resnet18)
    # input()
    p = calculate_trainable_parameters(resnet18)
    print(p)

    for mi, ens_grads in grads.items():
        if mi == 0:
            continue
        f = lambda x: torch.mean(x, 0)

        ens_grads = {name: f(torch.stack(gs, 0)).detach().cpu() for name, gs in ens_grads.items()}

        masks = get_masks_from_gradients(gradients=ens_grads, prune_percentage=0.3,
                                         global_pruning=False)

        m = extract_inner_model(resnet18, masks, False)
        print(m)
        p = calculate_trainable_parameters(m)
        print(p)

        print(m(x).shape)
        # break
