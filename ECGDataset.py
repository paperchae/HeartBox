import pandas as pd
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class ECGDataset(Dataset):
    def __init__(self, ecg, label, pid, device, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.ecg = ecg
        self.label = torch.FloatTensor(label)
        self.pid = pid
        self.length = len(ecg)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ecg = self.ecg[index]
        label = self.label[index]
        idx = self.pid[index]
        if self.transform is not None:
            ecg = self.transform(ecg)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return ecg, label, idx


class Compose:
    def __init__(self, transforms: list) -> None:
        """
        Args:
            transforms (list): list of transforms to compose.
        Returns:
            Composed transforms which sequentially perform a list of transforms.
        """
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class ToTensor:
    def __init__(self, dtype) -> None:
        """
        Args:
            dtype (str): data type of tensor to convert to.
        Returns:
            Tensor of specified type.
        """
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        # print('ToTensor')
        if self.dtype == 'float':
            x = torch.FloatTensor(x)
        elif self.dtype == 'long':
            x = torch.LongTensor(x)
        else:
            raise ValueError(f'Invalid type: {self.dtype}')
        return x


class Resample:
    def __init__(self, signal_time_length: int, sample_rate_to: int) -> None:
        """
        Args:
            signal_time_length (int): length of signal in seconds.
            sample_rate_to (int): sample rate to resample to.
        Returns:
            Resampled signal.
        """
        self.signal_time_length = signal_time_length
        self.sample_rate_to = sample_rate_to

    def __call__(self, x: Any) -> Any:
        # TODO: Implement Resample function which resamples signal to specified sample rate
        return NotImplemented


class Standardize:
    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        """
        Args:
            dim (int): dimension to compute mean and standard deviation over.
            eps (float): epsilon to add to standard deviation to prevent division by zero.
        Returns:
            Standardized signal.
        """
        self.dim = dim
        self.eps = eps

    def __call__(self, x: Any) -> Any:
        # TODO: Implement Standardize function which standardizes signal
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)
        # return NotImplemented


class RandomCrop:
    def __init__(self, ratio=0.8) -> None:
        """
        Args:
            ratio (): ratio of signal to crop.
        Returns:
            Randomly cropped signal.
        """
        self.ratio = ratio
        # assert 0 < ratio <= 1

    def __call__(self, x: Any) -> Any:
        # TODO: Implement RandomCrop function which randomly crops signal with specified size
        size = int(x.shape[-1] * self.ratio)
        offset = np.random.randint(0, x.shape[-1] - size)
        return x[offset:offset + size]

        # return NotImplemented


class RandomMask:
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, x: Any) -> Any:
        mask_size = int(x.shape[-1] * self.ratio)
        offset = np.random.randint(0, x.shape[-1] - mask_size)
        if np.random.rand() < 0.5:
            x[offset:offset + mask_size] = 0
        return x


class RandomVerticalFlip:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x: Any) -> Any:
        if np.random.rand() < self.ratio:
            x = -x + 1
        return x


class RandomHorizontalFlip:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x: Any) -> Any:
        # print('RandomHorizontalFlip')
        if np.random.rand() < self.ratio:
            x = torch.fliplr(x.view(1, -1)).view(-1)
        return x

class RandomNoise:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x: Any) -> Any:
        # print('RandomHorizontalFlip')
        if np.random.rand() < self.ratio:
            x = x + torch.randn(x.shape) * 0.01
        return x
# composed = Compose([ToTensor('float'), RandomCrop(0.7)])
# import numpy as np
#
# test = composed(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
# print(test)
