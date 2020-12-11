import os
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.models import vgg11
from torchvision.transforms import transforms

from models import LeNet, LeNet_300_100


class EarlyStopping:
    def __init__(self, tolerance, min=True, **kwargs):
        self.initial_tolerance = tolerance
        self.tolerance = tolerance
        if min:
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


# def path_exists(path):
#     return os.path.exists(path)


def ensures_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def save_model():
    pass


def load_model():
    pass


def get_model(name, input_size=None, output=None):
    name = name.lower()
    if name == 'lenet-300-100':
        return LeNet_300_100(input_size, output)
    elif name == 'lenet-5':
        return LeNet(input_size, output)
    elif 'vgg' in name:
        # if 'bn' in name:

        if name == 'vgg11':
            vgg = vgg11(pretrained=False, num_classes=output)
        else:
            assert False

        for n, m in vgg.named_modules():
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    if m.bias.sum() == 0:
                        m.bias = None
        return vgg
    else:
        assert False


def get_dataset(name):
    if name == 'mnist':
        t = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                            # nn.Flatten(0)
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
        input_size = 1

    elif name == 'flat_mnist':
        t = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                            # nn.Flatten(0)
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

    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='./datasets/cifar10', train=True, download=True, transform=train_transform)

        test_set = torchvision.datasets.CIFAR10(
            root='./datasets/cifar10', train=False, download=True, transform=transform)

        input_size, classes = 3, 10

    else:
        assert False

    return train_set, test_set, input_size, classes


def get_optimizer(optimizer: str, lr: float, momentum: float = 0, l1: float = 0, l2: float = 0, annealing=None):
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
                return StepLR(opt, step_size=annealing['step_size'], gamma=annealing.get('gamma', 0.1),
                              verbose=annealing.get('verbose', False))
            elif t == 'multi_step':
                return MultiStepLR(opt, milestones=annealing['milestones'], gamma=annealing.get('gamma', 0.1),
                                   verbose=annealing.get('verbose', False))
            else:
                assert False
        else:
            return None

    return opt, l1loss, scheduler


@torch.no_grad()
def eval_model(model, dataset, topk=None, device='cpu'):
    model.eval()
    predictions = []
    true = []

    with torch.no_grad():
        for x, y in dataset:
            true.extend(y.tolist())
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            top_classes = torch.topk(outputs, outputs.size(-1))[1]
            predictions.extend(top_classes.tolist())

    predictions = np.asarray(predictions)
    true = np.asarray(true)

    accuracies = accuracy_score(true, predictions, topk=topk)

    # cm = confusion_matrix(true, predictions)
    # TODO: aggiungere calcolo f1 score

    return accuracies


def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
    if topk is None:
        topk = [1, 5]

    if isinstance(topk, int):
        topk = [topk]

    assert len(expected) == len(predicted)
    assert predicted.shape[1] >= max(topk)

    res = defaultdict(int)
    total = len(expected)

    for t, p in zip(expected, predicted):
        for k in topk:
            if t in p[:k]:
                res[k] += 1

    res = {k: v / total for k, v in res.items()}

    return res


def confusion_matrix(true: np.asarray, pred: np.asarray, ):
    num_classes = np.max(true) + 1
    pred = pred[:, 0]
    m = np.zeros((num_classes, num_classes), dtype=int)
    for pred, exp in zip(pred, true):
        m[pred][exp] += 1
    return m
