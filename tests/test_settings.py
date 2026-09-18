import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import settings as S  # noqa: E402

ENV_KEYS = ["NUM_WORKERS", "USE_GPU", "NUM_EPOCHS", "GLOBAL_BATCH_SIZE", "SMOKE_TEST",
            "DATA_ROOT", "STORAGE_PATH", "MLFLOW_TRACKING_URI", "ANYSCALE_ARTIFACT_STORAGE", "LR"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(S, "CLUSTER_STORAGE", str(tmp_path / "no-such-mount"))
    monkeypatch.setattr(S, "LOCAL_STORAGE", str(tmp_path / "no-such-local"))
    yield


def test_defaults_without_any_shared_storage(capsys):
    s = S.Settings.from_env()
    assert s.num_workers == 2 and s.use_gpu is True and s.num_epochs == 2
    assert s.global_batch_size == 128 and s.smoke_test is False and s.subset_size is None
    assert s.storage_path == f"/tmp/{S.APP_NAME}"
    assert "WARNING" in capsys.readouterr().out


def test_smoke_test_forces_one_epoch_and_subset(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "1")
    monkeypatch.setenv("NUM_EPOCHS", "7")
    s = S.Settings.from_env()
    assert s.smoke_test is True and s.num_epochs == 1 and s.subset_size == 2048


def test_use_gpu_parsing(monkeypatch):
    for raw, want in [("0", False), ("false", False), ("no", False), ("1", True), ("true", True)]:
        monkeypatch.setenv("USE_GPU", raw)
        assert S.Settings.from_env().use_gpu is want


def test_cluster_storage_preferred(monkeypatch, tmp_path):
    mount = tmp_path / "cluster"
    mount.mkdir()
    monkeypatch.setattr(S, "CLUSTER_STORAGE", str(mount))
    assert S.resolve_storage_path(None) == f"{mount}/{S.APP_NAME}"


def test_artifact_storage_fallback(monkeypatch):
    monkeypatch.setenv("ANYSCALE_ARTIFACT_STORAGE", "s3://bucket/org/cloud/artifact_storage/")
    assert S.resolve_storage_path(None) == f"s3://bucket/org/cloud/artifact_storage/{S.APP_NAME}"


def test_artifact_sentinel(monkeypatch):
    monkeypatch.setenv("ANYSCALE_ARTIFACT_STORAGE", "s3://bucket/root")
    assert S.resolve_storage_path("artifact://demo/run") == "s3://bucket/root/demo/run"


def test_artifact_sentinel_without_env_raises():
    with pytest.raises(RuntimeError):
        S.resolve_storage_path("artifact://demo")


def test_explicit_path_expands_vars(monkeypatch):
    monkeypatch.setenv("MY_BUCKET", "s3://b")
    assert S.resolve_storage_path("$MY_BUCKET/x") == "s3://b/x"


def test_mlflow_uri_defaults_to_file_store_under_local_storage_path(monkeypatch, tmp_path):
    mount = tmp_path / "cluster"
    mount.mkdir()
    monkeypatch.setattr(S, "CLUSTER_STORAGE", str(mount))
    s = S.Settings.from_env()
    assert s.mlflow_tracking_uri == f"file://{mount}/{S.APP_NAME}/mlflow"


def test_mlflow_uri_falls_back_when_storage_is_remote(monkeypatch):
    monkeypatch.setenv("STORAGE_PATH", "s3://bucket/x")
    s = S.Settings.from_env()
    assert s.mlflow_tracking_uri == f"file:///tmp/{S.APP_NAME}/mlflow"


def test_train_loop_config_keys():
    cfg = S.Settings.from_env().as_train_loop_config()
    assert set(cfg) == {"num_epochs", "global_batch_size", "lr", "data_root", "subset_size"}


def test_ray_init_with_repo_ships_the_repo_as_working_dir(monkeypatch):
    """Every script calls this instead of bare ray.init() so Ray workers can `from src.x import y`.

    A Ray worker process does not inherit the driver's sys.path, so without a
    runtime_env working_dir, workers that land in a different process (or
    node) than the driver fail with ModuleNotFoundError: No module named 'src'.
    """
    import ray

    captured = {}
    monkeypatch.setattr(ray, "init", lambda **kwargs: captured.update(kwargs))

    S.ray_init_with_repo()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(S.__file__)))
    assert captured["runtime_env"]["working_dir"] == repo_root
    assert captured["runtime_env"]["excludes"] == [".git", "notebooks", "data", "**/__pycache__"]
