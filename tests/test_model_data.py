import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_data_loader, build_dataset, raw_mnist  # noqa: E402
from src.model import build_resnet18  # noqa: E402


def test_resnet18_accepts_grayscale_28x28():
    model = build_resnet18()
    out = model(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_subset_and_loader_shapes(tmp_path):
    ds = build_dataset(str(tmp_path), subset_size=64)
    assert len(ds) == 64
    loader = build_data_loader(str(tmp_path), batch_size=16, subset_size=64)
    images, labels = next(iter(loader))
    assert images.shape == (16, 1, 28, 28) and labels.shape == (16,)
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_raw_mnist_has_uint8_tensors(tmp_path):
    ds = raw_mnist(str(tmp_path))
    assert ds.data.dtype == torch.uint8 and ds.data.shape[1:] == (28, 28)
