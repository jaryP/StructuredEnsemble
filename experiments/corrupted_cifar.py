import os
from itertools import chain
from typing import Callable, Optional, Tuple, Any

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
from torchvision.transforms import transforms

from eval import get_logits, eval_method
from experiments.calibration import ece_score

BENCHMARK_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression',
]

EXTRA_CORRUPTIONS = [
    'gaussian_blur',
    'saturate',
    'spatter',
    'speckle_noise',
]


class CorruptedCifar10(VisionDataset):
    download_url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
    labels_filename = 'labels.npy'

    corruption_to_filename = {
        'gaussian_noise': 'gaussian_noise.npy',
        'shot_noise': 'shot_noise.npy',
        'impulse_noise': 'impulse_noise.npy',
        'defocus_blur': 'defocus_blur.npy',
        'frosted_glass_blur': 'glass_blur.npy',
        'motion_blur': 'motion_blur.npy',
        'zoom_blur': 'zoom_blur.npy',
        'snow': 'snow.npy',
        'frost': 'frost.npy',
        'fog': 'fog.npy',
        'brightness': 'brightness.npy',
        'contrast': 'contrast.npy',
        'elastic': 'elastic_transform.npy',
        'pixelate': 'pixelate.npy',
        'jpeg_compression': 'jpeg_compression.npy',
        'gaussian_blur': 'gaussian_blur.npy',
        'saturate': 'saturate.npy',
        'spatter': 'spatter.npy',
        'speckle_noise': 'speckle_noise.npy',
    }

    def __init__(self,
                 root: str,
                 corruption: str,
                 severity: int,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):

        super(CorruptedCifar10, self).__init__(root, transform=transform,
                                               target_transform=target_transform)

        assert severity <= 5, 'Severity must be in range [1, 5]'
        assert corruption in chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS), \
            'Item need to be one of {}'.format(chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS))

        if download:
            pass

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # Tensorflow-inspired code
        images_file = os.path.join(self.files_folder, F'{self.corruption_to_filename[corruption]}')
        labels_file = os.path.join(self.files_folder, F'{self.labels_filename}')

        images = np.load(images_file)
        labels = np.load(labels_file)

        num_images = labels.shape[0] // 5

        # Labels are stacked 5 times so we can just read the first iteration
        labels = labels[:num_images]
        images = images[(severity - 1) * num_images:severity * num_images]

        # images = np.transpose(images, (0, 3, 1, 2))
        self.data, self.targets = images, labels

    @property
    def folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def files_folder(self) -> str:
        return os.path.join(self.folder, 'CIFAR-10-C')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        for i in chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS):
            if not os.path.exists(os.path.join(self.files_folder, self.corruption_to_filename[i])):
                return False
        return True

    def download(self) -> None:
        """Download the Corrupted Cifar10 data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        filename = self.download_url.rpartition('/')[2]
        download_and_extract_archive(self.download_url, download_root=self.folder, filename=filename, md5=None)
        print('Done!')


class CorruptedCifar100(CorruptedCifar10):
    download_url = 'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar'
    labels_filename = 'labels.npy'

    corruption_to_filename = {
        'gaussian_noise': 'gaussian_noise.npy',
        'shot_noise': 'shot_noise.npy',
        'impulse_noise': 'impulse_noise.npy',
        'defocus_blur': 'defocus_blur.npy',
        'frosted_glass_blur': 'glass_blur.npy',
        'motion_blur': 'motion_blur.npy',
        'zoom_blur': 'zoom_blur.npy',
        'snow': 'snow.npy',
        'frost': 'frost.npy',
        'fog': 'fog.npy',
        'brightness': 'brightness.npy',
        'contrast': 'contrast.npy',
        'elastic': 'elastic_transform.npy',
        'pixelate': 'pixelate.npy',
        'jpeg_compression': 'jpeg_compression.npy',
        'gaussian_blur': 'gaussian_blur.npy',
        'saturate': 'saturate.npy',
        'spatter': 'spatter.npy',
        'speckle_noise': 'speckle_noise.npy',
    }

    def __init__(self,
                 root: str,
                 corruption: str,
                 severity: int,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        super(CorruptedCifar100, self).__init__(root, transform=transform,
                                                corruption=corruption, severity=severity,
                                                download=download,
                                                target_transform=target_transform)

    @property
    def files_folder(self) -> str:
        return os.path.join(self.folder, 'CIFAR-100-C')


def corrupted_cifar_uncertainty(method, batch_size, use_extra_corruptions=False, dataset='cifar10_vgg11'):
    assert dataset in ['cifar100', 'cifar10_vgg11']

    if dataset == 'cifar10_vgg11':
        dataset = CorruptedCifar10
        t = [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    else:
        dataset = CorruptedCifar100
        t = [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]

    if use_extra_corruptions:
        corruptions = chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS)
    else:
        corruptions = BENCHMARK_CORRUPTIONS

    scores = {name: {} for name in corruptions}
    entropy = {name: {} for name in corruptions}
    buc = {name: {} for name in corruptions}
    eces = {name: {} for name in corruptions}

    for name in corruptions:
        for severity in range(1, 6):
            # CIFAR 10
            loader = torch.utils.data.DataLoader(dataset=dataset('./datasets/', download=True,
                                                                          corruption=name, severity=severity,
                                                                          transform=transforms.Compose(t)),
                                                 batch_size=batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=4)

            ece, _, _, _ = ece_score(method, loader)

            eces[name][severity] = ece
            scores[name][severity] = eval_method(method, dataset=loader)

            logits, labels, predictions = get_logits(method, loader)

            _entropy = dataset_entropy(logits)
            _bcu = epistemic_aleatoric_uncertainty(logits)[0]
            # mask = labels == predictions
            # print(mask)
            # print(_bcu.shape, entropy.shape, mask.shape)

            entropy[name][severity] = np.mean(_entropy)
            buc[name][severity] = np.mean(_bcu)
            # print(labels == predictions)
            # print(name, severity)
            # print(scores[name][severity])
            # print(np.mean(_bcu[mask]), np.mean(_bcu[~mask]))
            # print(np.mean(_entropy[mask]), np.mean(_entropy[~mask]))
            # print(np.mean(bcu[mask]), np.mean(buc[~mask]))
            # print(np.mean(entropy[mask]), np.mean(entropy[~mask]))

    return entropy, scores, buc, ece


def dataset_entropy(logits):
    # TODO: Implementae mia incertezza
    with torch.no_grad():
        # probs, true = get_logits(method, dataset)
        logits = logits.mean(1)
        classes = logits.shape[-1]

        softmax = np.exp(logits)
        softmax = softmax / softmax.sum(1, keepdims=True)

        entropy = (softmax * np.log(softmax + 1e-12)) / np.log(classes)
        entropy = -entropy.sum(1)  # / np.log(classes)
        # entropy = np.mean(entropy)

    return entropy


def det(x):
    classes = x.shape[-1]

    mn = 1 / (classes ** classes)
    mx = mn * (2 ** (classes - 1))

    det = np.linalg.det(x + (np.eye(classes) / classes))
    det = (det - mn) / (mx - mn)

    return det


def epistemic_aleatoric_uncertainty(logits):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)

    softmax = np.exp(logits)
    softmax = softmax / softmax.sum(2, keepdims=True)
    p_hat = np.mean(softmax, 1)

    # p = torch.softmax(, 2)
    #
    # p = p.detach().cpu().numpy()
    # p = np.transpose(p, (1, 0, 2))
    #
    # p_hat = p_hat.detach().cpu().numpy()

    t = softmax.shape[1]
    classes = softmax.shape[-1]

    determinants = []
    variances = []

    mn = classes ** -classes
    mx = mn * np.power(2, classes - 1)

    for _bi in range(softmax.shape[0]):
        _bp = softmax[_bi]
        _bp_hat = p_hat[_bi]

        al = np.zeros((classes, classes))
        ep = np.zeros((classes, classes))

        for i in range(t):
            _p = _bp[i]
            aleatoric = np.diag(_p) - np.outer(_p, _p)
            al += aleatoric
            d = _p - _bp_hat
            epistemic = np.outer(d, d)
            ep += epistemic

        al /= t
        ep /= t

        var = al + ep

        variances.append(var)

        det = np.linalg.det(var + (np.eye(classes) / classes))
        det = (det - mn) / (mx - mn)

        determinants.append(det)

    determinants = np.asarray(determinants)
    variances = np.asarray(variances)

    return determinants, variances
