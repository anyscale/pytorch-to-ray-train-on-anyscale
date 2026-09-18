import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_ray_train import assert_shard_has_a_batch, unwrap  # noqa: E402


def test_assert_shard_has_a_batch_passes_when_worker_shard_covers_one_batch():
    # 64 samples, 2 workers -> 32 per worker, which covers a batch of 16.
    assert_shard_has_a_batch(dataset_size=64, world_size=2, per_worker_batch=16)


def test_assert_shard_has_a_batch_raises_on_the_per_worker_case_drop_last_missed():
    """The whole-dataset size can look fine while the per-worker shard is too small.

    64 samples split across 8 workers leaves 8 samples per worker, smaller than a
    batch of 16, even though 64 >= 16 (the check drop_last used to make on the
    undivided dataset, before DistributedSampler shards it per worker).
    """
    with pytest.raises(AssertionError, match="64.*8 workers|8 workers.*64|dataset/subset size 64"):
        assert_shard_has_a_batch(dataset_size=64, world_size=8, per_worker_batch=16)


class _FakeDDPModule:
    """Stands in for a DistributedDataParallel-wrapped model: has a `.module`."""

    def __init__(self, module):
        self.module = module


def test_unwrap_returns_the_inner_module_when_wrapped_in_ddp():
    inner = object()
    assert unwrap(_FakeDDPModule(inner)) is inner


def test_unwrap_returns_the_model_itself_when_never_wrapped():
    # prepare_model only wraps in DDP when world_size > 1 (single-worker runs stay unwrapped).
    model = object()
    assert unwrap(model) is model
