"""Environment-driven settings shared by every notebook and script.

Set these variables before launching a notebook or script. All have defaults
that work on an Anyscale workspace with GPU workers.

    NUM_WORKERS        Ray Train workers (default 2)
    USE_GPU            1 or 0 (default 1)
    NUM_EPOCHS         training epochs (default 2; forced to 1 when SMOKE_TEST=1)
    GLOBAL_BATCH_SIZE  split across workers (default 128)
    LR                 learning rate (default 1e-3)
    SMOKE_TEST         1 uses a 2,048-sample subset and 1 epoch (default 0)
    DATA_ROOT          torchvision download root, per node (default /mnt/local_storage/data)
    STORAGE_PATH       Ray Train RunConfig.storage_path; see resolve_storage_path()
    MLFLOW_TRACKING_URI  appendix A2 only
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

APP_NAME = "pytorch-to-ray-train"
CLUSTER_STORAGE = "/mnt/cluster_storage"
LOCAL_STORAGE = "/mnt/local_storage"
ARTIFACT_PREFIX = "artifact://"
SMOKE_SUBSET = 2048


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_storage_path(explicit: str | None) -> str:
    """Pick a storage path every worker node can reach.

    Order: explicit value (with ``artifact://`` sentinel or ``$VAR`` expansion),
    shared cluster storage, Anyscale artifact storage, then /tmp with a warning.
    """
    if explicit:
        if explicit.startswith(ARTIFACT_PREFIX):
            base = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
            if not base:
                raise RuntimeError(
                    "STORAGE_PATH uses artifact:// but ANYSCALE_ARTIFACT_STORAGE is not set. "
                    "Run inside an Anyscale workspace or job, or give a full path."
                )
            sub = explicit[len(ARTIFACT_PREFIX):].strip("/")
            return f"{base.rstrip('/')}/{sub}"
        return os.path.expandvars(explicit)
    if os.path.isdir(CLUSTER_STORAGE):
        return f"{CLUSTER_STORAGE}/{APP_NAME}"
    base = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
    if base:
        return f"{base.rstrip('/')}/{APP_NAME}"
    print(
        f"WARNING: no shared storage found. Using /tmp/{APP_NAME}, which only works "
        "when every Ray Train worker runs on this node."
    )
    return f"/tmp/{APP_NAME}"


def _default_data_root() -> str:
    return f"{LOCAL_STORAGE}/data" if os.path.isdir(LOCAL_STORAGE) else "/tmp/data"


@dataclass
class Settings:
    num_workers: int
    use_gpu: bool
    num_epochs: int
    global_batch_size: int
    smoke_test: bool
    data_root: str
    storage_path: str
    lr: float

    @classmethod
    def from_env(cls) -> "Settings":
        smoke = _env_bool("SMOKE_TEST", False)
        return cls(
            num_workers=int(os.environ.get("NUM_WORKERS", "2")),
            use_gpu=_env_bool("USE_GPU", True),
            num_epochs=1 if smoke else int(os.environ.get("NUM_EPOCHS", "2")),
            global_batch_size=int(os.environ.get("GLOBAL_BATCH_SIZE", "128")),
            smoke_test=smoke,
            data_root=os.environ.get("DATA_ROOT", _default_data_root()),
            storage_path=resolve_storage_path(os.environ.get("STORAGE_PATH")),
            lr=float(os.environ.get("LR", "1e-3")),
        )

    @property
    def subset_size(self) -> int | None:
        return SMOKE_SUBSET if self.smoke_test else None

    @property
    def local_output_dir(self) -> str:
        base = LOCAL_STORAGE if os.path.isdir(LOCAL_STORAGE) else "/tmp"
        return f"{base}/{APP_NAME}"

    @property
    def mlflow_tracking_uri(self) -> str:
        explicit = os.environ.get("MLFLOW_TRACKING_URI")
        if explicit:
            return explicit
        if self.storage_path.startswith("/"):
            return f"file://{self.storage_path}/mlflow"
        return f"file:///tmp/{APP_NAME}/mlflow"

    def as_train_loop_config(self) -> dict:
        """The hyperparameters a Ray Train worker needs. Keep it small and picklable."""
        return {
            "num_epochs": self.num_epochs,
            "global_batch_size": self.global_batch_size,
            "lr": self.lr,
            "data_root": self.data_root,
            "subset_size": self.subset_size,
        }

    def describe(self) -> None:
        print("Settings (override with environment variables, see src/settings.py):")
        for key, value in asdict(self).items():
            print(f"  {key:18} = {value}")
        print(f"  {'subset_size':18} = {self.subset_size}")
        try:
            import ray

            if ray.is_initialized():
                res = ray.cluster_resources()
                print(f"Ray cluster right now: CPU={res.get('CPU', 0):.0f} GPU={res.get('GPU', 0):.0f} "
                      f"nodes={sum(1 for k in res if k.startswith('node:') and '__internal' not in k)}")
                print("  (Anyscale adds nodes on demand, so GPU can be 0 until a workload asks for them.)")
        except ImportError:
            pass


def ray_init_with_repo() -> None:
    """ray.init(), but with the repo shipped to every worker as a runtime_env working_dir.

    A Ray worker process does not inherit the driver's sys.path, so a bare
    ray.init() leaves `from src.xxx import yyy` failing inside worker
    processes with "ModuleNotFoundError: No module named 'src'" as soon as a
    worker lands on a different process (or node) than the driver. Every
    script that talks to Ray imports this instead of calling ray.init()
    directly, so the fix lives in one place.
    """
    import ray

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ray.init(
        runtime_env={
            "working_dir": repo_root,
            "excludes": [".git", "notebooks", "data", "**/__pycache__"],
        }
    )
