import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tune_ray_train import best_checkpoint  # noqa: E402


class _FakeResult:
    def __init__(self, metrics: dict):
        self.metrics = metrics


class _FakeResultGrid:
    """Just enough of ray.tune.ResultGrid for best_checkpoint(): get_best_result()."""

    def __init__(self, best_result: _FakeResult):
        self._best_result = best_result

    def get_best_result(self):
        return self._best_result


def test_best_checkpoint_sets_filesystem_for_a_cloud_style_storage_path():
    """TuneReportCallback reports Checkpoint.path scheme-stripped on cloud storage.

    Constructing Checkpoint(path=that_string) with no filesystem asks pyarrow to
    infer one from the path itself, and a scheme-less cloud path has nothing to
    infer from ("URI has empty scheme"). best_checkpoint() must derive the
    filesystem from storage_path (which still has its scheme) and attach it
    explicitly, rather than doing string surgery on the checkpoint path.
    """
    scheme_less_checkpoint_path = (
        "bucket/pytorch-to-ray-train/jobs/train-trial_id=d8e09_00002/checkpoint_2026-09-17_17-20-52.246322"
    )
    results = _FakeResultGrid(_FakeResult({"checkpoint_path": scheme_less_checkpoint_path}))

    checkpoint = best_checkpoint(results, "s3://bucket/pytorch-to-ray-train")

    assert checkpoint.path == scheme_less_checkpoint_path
    assert checkpoint.filesystem is not None
    assert type(checkpoint.filesystem).__name__ == "S3FileSystem"


def test_best_checkpoint_still_works_for_local_storage_path(tmp_path):
    """Local storage needs no scheme either way; this is the path the old code covered."""
    checkpoint_dir = tmp_path / "checkpoint_0"
    results = _FakeResultGrid(_FakeResult({"checkpoint_path": str(checkpoint_dir)}))

    checkpoint = best_checkpoint(results, str(tmp_path))

    assert checkpoint.path == str(checkpoint_dir)
    assert type(checkpoint.filesystem).__name__ == "LocalFileSystem"
