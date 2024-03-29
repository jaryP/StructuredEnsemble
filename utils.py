import os

import numpy as np
import torch
import torchvision
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets
from torchvision.models import vgg11, AlexNet, vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from eval import eval_model
from models import LeNet, LeNet_300_100, resnet20, resnet32


class EarlyStopping:
    def __init__(self, tolerance, min=True, **kwargs):
        self.initial_tolerance = tolerance
        self.tolerance = tolerance
        self.min = min
        if self.min:
            self.current_value = np.inf
            self.c = lambda a, b: a < b
        else:
            self.current_value = -np.inf
            self.c = lambda a, b: a > b

    def step(self, v):
        if self.c(v, self.current_value):
            self.tolerance = self.initial_tolerance
            self.current_value = v
            return 1
        else:
            self.tolerance -= 1
            if self.tolerance <= 0:
                return -1
        return 0

    def reset(self):
        self.tolerance = self.initial_tolerance
        self.current_value = 0
        if self.min:
            self.current_value = np.inf
            self.c = lambda a, b: a < b
        else:
            self.current_value = -np.inf
            self.c = lambda a, b: a > b


def ensures_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def get_model(name, input_size=None, output=None):
    name = name.lower()
    if name == 'lenet-300-100':
        model = LeNet_300_100(input_size, output)
    elif name == 'lenet-5':
        model = LeNet(input_size, output)
    elif 'vgg' in name:
        # if 'bn' in name:
        if name == 'vgg11':
            model = vgg11(pretrained=False, num_classes=output)
        elif name == 'vgg16':
            model = vgg16(pretrained=False, num_classes=output)
        else:
            assert False

        for n, m in model.named_modules():
            if hasattr(m, 'bias') and not isinstance(m, _BatchNorm):
                if m.bias is not None:
                    if m.bias.sum() == 0:
                        m.bias = None

    elif 'alexnet' in name:
        model = AlexNet(num_classes=output)

        for n, m in model.named_modules():
            if hasattr(m, 'bias') and not isinstance(m, _BatchNorm):
                if m.bias is not None:
                    if m.bias.sum() == 0:
                        m.bias = None
    elif 'resnet' in name:
        if name == 'resnet20':
            model = resnet20(num_classes=output)
        elif name == 'resnet32':
            model = resnet32(num_classes=output)
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


def get_dataset(name, model_name):
    if name == 'mnist':
        t = [torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = torchvision.transforms.Compose(t)

        train_set = torchvision.datasets.MNIST(
            root='./datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = torchvision.datasets.MNIST(
            root='./datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = 1

    elif name == 'flat_mnist':
        t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            torch.nn.Flatten(0)
                                            ])

        train_set = torchvision.datasets.MNIST(
            root='./datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = torchvision.datasets.MNIST(
            root='./datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = 28 * 28

    elif name == 'svhn':
        tt = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                 (0.19803012, 0.20101562, 0.19703614))]

        t = [
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                 (0.19803012, 0.20101562, 0.19703614))]

        # if 'resnet' in model_name:
        #     tt = [transforms.Resize(256), transforms.CenterCrop(224)] + tt
        #     t = [transforms.Resize(256), transforms.CenterCrop(224)] + t

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        train_set = torchvision.datasets.SVHN(
            root='./datasets/svhn', split='train', download=True,
            transform=train_transform)

        test_set = torchvision.datasets.SVHN(
            root='./datasets/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = 3, 10

    elif name == 'cifar10':
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

        train_set = torchvision.datasets.CIFAR10(
            root='./datasets/cifar10', train=True, download=True,
            transform=train_transform)

        test_set = torchvision.datasets.CIFAR10(
            root='./datasets/cifar10', train=False, download=True,
            transform=transform)

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

        train_set = torchvision.datasets.CIFAR100(
            root='./datasets/cifar100', train=True, download=True,
            transform=train_transform)

        test_set = torchvision.datasets.CIFAR100(
            root='./datasets/cifar100', train=False, download=True,
            transform=transform)

        input_size, classes = 3, 100
    elif name == 'tinyimagenet':
        tt = [
            transforms.ToTensor(),
            # transforms.RandomCrop(56),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262))
        ]

        t = [
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262))
        ]

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        # train_set = TinyImageNet(
        #     root='./datasets/tiny-imagenet-200', split='train',
        #     transform=transform)

        train_set = datasets.ImageFolder('./datasets/tiny-imagenet-200/train',
                                         transform=train_transform)

        # for x, y in train_set:
        #     if x.shape[0] == 1:
        #         print(x.shape[0] == 1)

        # test_set = TinyImageNet(
        #     root='./datasets/tiny-imagenet-200', split='val',
        #     transform=train_transform)
        test_set = datasets.ImageFolder('./datasets/tiny-imagenet-200/val',
                                        transform=transform)

        # for x, y in test_set:
        #     if x.shape[0] == 1:
        #         print(x.shape[0] == 1)

        input_size, classes = 3, 200

    else:
        assert False

    return train_set, test_set, input_size, classes


def get_optimizer(optimizer: str, lr: float,
                  momentum: float = 0,
                  l1: float = 0,
                  l2: float = 0,
                  annealing=None):
    def l1loss(network):
        l1_loss = 0
        if l1 <= 0:
            for m, n in network.named_modulues():
                l1_loss += torch.norm(n.weight.data, p=1)
            l1_loss *= l1
        return l1_loss

    optimizer = optimizer.lower()

    def opt(parameters):
        if optimizer == 'adam':
            return Adam(parameters, weight_decay=l2, lr=lr)
        elif optimizer == 'sgd':
            return SGD(parameters, momentum=momentum, weight_decay=l2, lr=lr)
        else:
            assert False

    # TODO: implement scheduler
    # scheduler = None

    def scheduler(opt):
        if annealing is not None:
            t = annealing['type'].lower()
            if t == 'step':
                return StepLR(opt, step_size=annealing['step_size'],
                              gamma=annealing.get('gamma', 0.1),
                              verbose=annealing.get('verbose', False))
            elif t == 'multi_step':
                return MultiStepLR(opt, milestones=annealing['milestones'],
                                   gamma=annealing.get('gamma', 0.1),
                                   verbose=annealing.get('verbose', False))
            else:
                assert False
        else:
            return None

    return opt, l1loss, scheduler


def calculate_trainable_parameters(model):
    params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            params += p.numel()
    return params


def train_model(model, optimizer, train_loader, epochs, scheduler,
                early_stopping=None,
                test_loader=None, eval_loader=None, device='cpu', t=1):
    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0
    model.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if t > 1:
                pred = torch.stack([model(x) for _ in range(t)], dim=0)
                pred = pred.mean(0)
            else:
                pred = model(x)
            loss = torch.nn.functional.cross_entropy(pred, y, reduction='none')
            losses.extend(loss.tolist())
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            eval_scores, _ = eval_model(model, eval_loader, topk=[1, 5],
                                        device=device)
        else:
            eval_scores = 0

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores[1]) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = model.state_dict()
                best_model_i = epoch
        else:
            best_model = model.state_dict()
            best_model_i = epoch

        train_scores, _ = eval_model(model, train_loader, device=device)
        test_scores, _ = eval_model(model, test_loader, device=device)

        bar.set_postfix(
            {'Train score': train_scores[1], 'Test score': test_scores[1],
             'Eval score': eval_scores[1] if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return best_model, scores, scores[best_model_i], mean_losses
