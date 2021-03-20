import torch
import torchvision
from continual_learning.backbone_networks import LeNet_300_100, LeNet, vgg11, \
    AlexNet, resnet20
from continual_learning.benchmarks import MNIST, CIFAR10, CIFAR100
from continual_learning.methods import Naive
from continual_learning.methods.task_incremental.multi_task.gg import \
    SingleTask, Pruning, SupermaskSuperposition, SuperMask, BatchEnsemble
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.transforms import transforms


def get_cl_dataset(name, model_name):
    if name == 'mnist':
        t = [torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = torchvision.transforms.Compose(t)

        dataset = MNIST(
            data_folder='./datasets_cl/mnist/',
            download_if_missing=True,
            transformer=t)

        classes = 10
        input_size = 1

    # elif name == 'svhn':
    #     tt = [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))]
    #
    #     t = [
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))]
    #
    #     # if 'resnet' in model_name:
    #     #     tt = [transforms.Resize(256), transforms.CenterCrop(224)] + tt
    #     #     t = [transforms.Resize(256), transforms.CenterCrop(224)] + t
    #
    #     transform = transforms.Compose(t)
    #     train_transform = transforms.Compose(tt)
    #
    #     train_set = torchvision.datasets.SVHN(
    #         root='./datasets/svhn', split='train', download=True, transform=train_transform)
    #
    #     test_set = torchvision.datasets.SVHN(
    #         root='./datasets/svhn', split='test', download=True, transform=transform)
    #
    #     input_size, classes = 3, 10

    elif name == 'cifar10':
        tt = [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))]

        t = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))]

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        dataset = CIFAR10(
            data_folder='./datasets_cl/cifar10',
            download_if_missing=True,
            transformer=train_transform,
            test_transformer=transform)

        input_size, classes = 3, 10

    elif name == 'cifar100':
        tt = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))]

        t = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))]

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        dataset = CIFAR100(
            data_folder='./datasets_cl/cifar10',
            download_if_missing=True,
            transformer=train_transform, test_transformer=transform)

        input_size, classes = 3, 100

    else:
        assert False

    return dataset, input_size, classes


def get_cl_model(name, input_size=None):
    name = name.lower()
    if name == 'lenet-300-100':
        model = LeNet_300_100(input_size)
    elif name == 'lenet-5':
        model = LeNet(input_size)
    elif 'vgg' in name:
        # if 'bn' in name:
        if name == 'vgg11':
            model = vgg11(pretrained=False)
        else:
            assert False

        for n, m in model.named_modules():
            if hasattr(m, 'bias') and not isinstance(m, _BatchNorm):
                if m.bias is not None:
                    if m.bias.sum() == 0:
                        m.bias = None

    elif 'alexnet' in name:
        model = AlexNet()

        for n, m in model.named_modules():
            if hasattr(m, 'bias') and not isinstance(m, _BatchNorm):
                if m.bias is not None:
                    if m.bias.sum() == 0:
                        m.bias = None
    elif 'resnet' in name:
        if name == 'resnet20':
            model = resnet20()
        else:
            assert False

        for n, m in model.named_modules():
            if hasattr(m, 'bias') and not isinstance(m, _BatchNorm):
                if m.bias is not None:
                    if m.bias.sum() == 0:
                        m.bias = None

    else:
        assert False

    return model


def get_cl_method(name, backbone, parameters, device):
    name = name.lower()

    if name == 'ensemble':
        return SingleTask()
    elif name == 'batch_ensemble':
        return BatchEnsemble(backbone)
    elif name == 'naive':
        return Naive()
    elif name == 'pruning':
        return Pruning(backbone=backbone, **parameters)
    elif name == 'supsup':
        return SupermaskSuperposition(backbone=backbone, **parameters)
    elif name == 'supermask':
        return SuperMask(device=device, **parameters)
