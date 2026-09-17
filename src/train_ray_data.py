"""Stage: the Ray Train loop fed by Ray Data instead of a DataLoader.

    python src/train_ray_data.py
    USE_GPU=0 NUM_WORKERS=2 SMOKE_TEST=1 python src/train_ray_data.py

Docs: https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html
"""
from __future__ import annotations

import os
import sys

import pyarrow as pa
import torch
from filelock import FileLock
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import ray
import ray.data
import ray.train
import ray.train.torch
from ray.data.extensions import ArrowTensorArray
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import raw_mnist  # noqa: E402
from src.model import build_resnet18  # noqa: E402
from src.settings import Settings, ray_init_with_repo  # noqa: E402
from src.train_ray_train import (  # noqa: E402
    default_run_name,
    load_checkpoint_state,
    maybe_fail_once,
    report_checkpoint,
)


def materialize_parquet(settings: Settings) -> str:
    """Write the MNIST train split to Parquet once; later runs reuse the same files.

    Every worker on every node calls this, so the write is guarded with a
    file lock and skipped once the target directory already has data.
    """
    path = os.path.join(settings.storage_path, "mnist_parquet", "train")
    os.makedirs(settings.data_root, exist_ok=True)
    with FileLock(os.path.join(settings.data_root, ".mnist_parquet.lock")):
        if os.path.isdir(path) and os.listdir(path):
            return path
        raw = raw_mnist(settings.data_root, train=True)
        images = raw.data.numpy()  # (N, 28, 28) uint8
        labels = raw.targets.numpy()  # (N,) int64
        table = pa.table({"image": ArrowTensorArray.from_numpy(images), "label": pa.array(labels)})
        ray.data.from_arrow(table).write_parquet(path)
    return path


def build_dataset(settings: Settings) -> ray.data.Dataset:
    """Read the materialized Parquet file back as a Ray Dataset."""
    return ray.data.read_parquet(materialize_parquet(settings))


def normalize(batch: dict) -> dict:
    """The Ray Data equivalent of ToTensor() then Normalize((0.5,), (0.5,))."""
    images = batch["image"].astype("float32") / 255.0
    images = (images - 0.5) / 0.5
    batch["image"] = images[:, None, :, :]  # (B, 28, 28) -> (B, 1, 28, 28)
    return batch


def train_loop_ray_data(config: dict) -> None:
    ctx = ray.train.get_context()
    world_size, rank = ctx.get_world_size(), ctx.get_world_rank()
    print(f"[rank {rank}/{world_size}] node_rank={ctx.get_node_rank()} host={os.uname().nodename}")

    model = ray.train.torch.prepare_model(build_resnet18())  # device placement + DistributedDataParallel
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    per_worker_batch = config["global_batch_size"] // world_size
    train_shard = ray.train.get_dataset_shard("train")  # Ray Data streams and shards the split itself

    start_epoch = load_checkpoint_state(model, optimizer)
    for epoch in range(start_epoch, config["num_epochs"]):
        maybe_fail_once(config, epoch)
        loss = torch.tensor(0.0)
        for batch in train_shard.iter_torch_batches(  # dict batches, no DataLoader, no .to(device)
            batch_size=per_worker_batch, prefetch_batches=2, dtypes={"image": torch.float32, "label": torch.int64}
        ):
            outputs = model(batch["image"])
            loss = criterion(outputs, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = {"loss": loss.item(), "epoch": epoch}
        if rank == 0:
            print(metrics)
        report_checkpoint(model, optimizer, metrics, epoch)


def main() -> None:
    settings = Settings.from_env()
    if not ray.is_initialized():
        ray_init_with_repo()
    settings.describe()

    train_ds = build_dataset(settings).map_batches(normalize, batch_format="numpy")
    if settings.subset_size:
        train_ds = train_ds.limit(settings.subset_size)

    run_name = os.environ.get("RUN_NAME") or default_run_name()
    print(f"run_name={run_name}  (export RUN_NAME={run_name} to resume this run later)")

    trainer = TorchTrainer(
        train_loop_ray_data,
        train_loop_config=settings.as_train_loop_config(),
        datasets={"train": train_ds},
        scaling_config=ScalingConfig(num_workers=settings.num_workers, use_gpu=settings.use_gpu),
        run_config=RunConfig(
            name=run_name,
            storage_path=settings.storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=2),
            failure_config=FailureConfig(max_failures=2),
        ),
    )
    result = trainer.fit()
    print("final metrics:", result.metrics)
    print("checkpoint:", result.checkpoint)
    print("run directory:", result.path)


if __name__ == "__main__":
    main()
