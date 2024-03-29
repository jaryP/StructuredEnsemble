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

        assert 1 <= severity <= 5, 'Severity must be in range [1, 5]'
        assert corruption in chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS), \
            'Item need to be one of {}'.format(
                chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS))

        if download:
            pass

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # Tensorflow-inspired code
        images_file = os.path.join(self.files_folder,
                                   F'{self.corruption_to_filename[corruption]}')
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
            if not os.path.exists(os.path.join(self.files_folder,
                                               self.corruption_to_filename[i])):
                return False
        return True

    def download(self) -> None:
        """Download the Corrupted Cifar10 data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        filename = self.download_url.rpartition('/')[2]
        download_and_extract_archive(self.download_url,
                                     download_root=self.folder,
                                     filename=filename, md5=None)
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
                                                corruption=corruption,
                                                severity=severity,
                                                download=download,
                                                target_transform=target_transform)

    @property
    def files_folder(self) -> str:
        return os.path.join(self.folder, 'CIFAR-100-C')


def corrupted_cifar_uncertainty(method, batch_size, use_extra_corruptions=False,
                                dataset='cifar10', normalize=False):
    assert dataset in ['cifar100', 'cifar10']

    if dataset == 'cifar10':
        dataset = CorruptedCifar10
        t = [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010))]
    else:
        dataset = CorruptedCifar100
        t = [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))]

    if use_extra_corruptions:
        corruptions = chain(BENCHMARK_CORRUPTIONS, EXTRA_CORRUPTIONS)
    else:
        corruptions = BENCHMARK_CORRUPTIONS

    scores = {name: {} for name in corruptions}
    entropy = {name: {} for name in corruptions}
    preds = {name: {} for name in corruptions}
    eces = {name: {} for name in corruptions}

    true_labels = None

    for name in corruptions:
        for severity in range(1, 6):
            # CIFAR 10
            loader = torch.utils.data.DataLoader(dataset=
                                                 dataset('./datasets/',
                                                         download=True,
                                                         corruption=name,
                                                         severity=severity,
                                                         transform=
                                                         transforms.Compose(t)),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True, num_workers=4)

            ece, _, _, _ = ece_score(method, loader)

            eces[name][severity] = ece
            scores[name][severity] = eval_method(method, dataset=loader)

            probs, labels, predictions = get_logits(method, loader)
            if true_labels is None:
                true_labels = labels

            hs = []
            for x, y in loader:
                # true.extend(y.tolist())
                p, _ = method.predict_proba(x, y, True)

                plog = (p + 1e-12).log()
                h = plog * p
                if normalize:
                    h = h / np.log(h.shape[-1])
                h = -torch.sum(h, -1)
                hs.extend(h.tolist())

            entropy[name][severity] = hs
            preds[name][severity] = predictions

    return entropy, preds, scores, true_labels


def dataset_entropy(logits):
    with torch.no_grad():
        logits = logits.mean(0)
        classes = logits.shape[-1]
        """
        plog = (p + 1e-12).log()
        h = plog * p
        if normalize:
            h = h / np.log(h.shape[-1])
        h = -torch.sum(h, -1)
        """
        exp = np.exp(logits)
        softmax = exp / exp.sum(-1, keepdims=True)
        entropy = (softmax * np.log(softmax + 1e-12)) / np.log(classes)
        entropy = -entropy.sum(-1)

    return entropy
