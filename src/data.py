"""MNIST dataset and DataLoader builders.

Every Ray Train worker calls these, possibly several workers on one node at
the same time, so the download is guarded with a file lock.
"""
from __future__ import annotations

import os

from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

MNIST_TRANSFORM = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


def raw_mnist(data_root: str, train: bool = True) -> MNIST:
    """MNIST without transforms: PIL images and uint8 tensors, for plotting and export."""
    os.makedirs(data_root, exist_ok=True)
    with FileLock(os.path.join(data_root, ".mnist.lock")):
        return MNIST(root=data_root, train=train, download=True)


def build_dataset(data_root: str, train: bool = True, subset_size: int | None = None, transform=MNIST_TRANSFORM):
    os.makedirs(data_root, exist_ok=True)
    with FileLock(os.path.join(data_root, ".mnist.lock")):
        dataset = MNIST(root=data_root, train=train, download=True, transform=transform)
    if subset_size:
        dataset = Subset(dataset, range(subset_size))
    return dataset


def build_data_loader(data_root: str, batch_size: int, subset_size: int | None = None, shuffle: bool = True) -> DataLoader:
    dataset = build_dataset(data_root, subset_size=subset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
