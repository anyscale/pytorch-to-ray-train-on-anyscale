"""Stage 02: the same training loop, distributed with Ray Train v2.

    python src/train_ray_train.py                             # NUM_WORKERS workers on GPUs
    USE_GPU=0 NUM_WORKERS=2 SMOKE_TEST=1 python src/train_ray_train.py
    RUN_NAME=my-run python src/train_ray_train.py             # same name + same storage path = resume

Docs: https://docs.ray.io/en/latest/train/getting-started-pytorch.html
"""
from __future__ import annotations

import datetime
import os
import sys
import tempfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import ray
import ray.train
import ray.train.torch
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_data_loader  # noqa: E402
from src.model import build_resnet18  # noqa: E402
from src.settings import Settings, ray_init_with_repo  # noqa: E402


def load_checkpoint_state(model, optimizer) -> int:
    """Restore model, optimizer, and epoch from the latest reported checkpoint.

    Returns the epoch to start from (0 when there is nothing to restore).
    """
    checkpoint = ray.train.get_checkpoint()
    if not checkpoint:
        return 0
    with checkpoint.as_directory() as ckpt_dir:
        model.module.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu"))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu"))
        start_epoch = torch.load(os.path.join(ckpt_dir, "extra_state.pt"))["epoch"] + 1
    print(f"[rank {ray.train.get_context().get_world_rank()}] resuming from epoch {start_epoch}")
    return start_epoch


def report_checkpoint(model, optimizer, metrics: dict, epoch: int) -> None:
    """Every rank calls ray.train.report (it is a barrier). Only rank 0 attaches a checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = None
        if ray.train.get_context().get_world_rank() == 0:
            torch.save(model.module.state_dict(), os.path.join(tmp, "model.pt"))  # .module unwraps DDP
            torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
            torch.save({"epoch": epoch}, os.path.join(tmp, "extra_state.pt"))
            checkpoint = Checkpoint.from_directory(tmp)
        ray.train.report(metrics, checkpoint=checkpoint)


def maybe_fail_once(config: dict, epoch: int) -> None:
    """Teaching aid for notebook 02: crash the last worker once at epoch 1 so FailureConfig recovers."""
    if not config.get("fail_once") or epoch != 1:
        return
    ctx = ray.train.get_context()
    if ctx.get_world_rank() != ctx.get_world_size() - 1:
        return
    marker = os.path.join(config["fail_marker_dir"], "already_failed")
    if os.path.exists(marker):
        return
    os.makedirs(config["fail_marker_dir"], exist_ok=True)
    open(marker, "w").close()
    raise RuntimeError("Simulated worker failure, on purpose, to show automatic recovery")


def train_loop_per_worker(config: dict) -> None:
    ctx = ray.train.get_context()
    world_size, rank = ctx.get_world_size(), ctx.get_world_rank()
    print(f"[rank {rank}/{world_size}] node_rank={ctx.get_node_rank()} host={os.uname().nodename}")

    model = ray.train.torch.prepare_model(build_resnet18())  # device placement + DistributedDataParallel
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    per_worker_batch = config["global_batch_size"] // world_size
    data_loader = ray.train.torch.prepare_data_loader(  # DistributedSampler + batches moved to the device
        build_data_loader(config["data_root"], per_worker_batch, config["subset_size"])
    )

    start_epoch = load_checkpoint_state(model, optimizer)
    for epoch in range(start_epoch, config["num_epochs"]):
        maybe_fail_once(config, epoch)
        if world_size > 1:
            data_loader.sampler.set_epoch(epoch)
        loss = torch.tensor(0.0)
        for images, labels in data_loader:  # no .to(device): prepare_data_loader did it
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = {"loss": loss.item(), "epoch": epoch}
        if rank == 0:
            print(metrics)
        report_checkpoint(model, optimizer, metrics, epoch)


def build_trainer(settings: Settings, run_name: str, fail_once: bool = False) -> TorchTrainer:
    train_loop_config = settings.as_train_loop_config()
    if fail_once:
        train_loop_config.update(
            fail_once=True, fail_marker_dir=os.path.join(settings.storage_path, "markers", run_name)
        )
    return TorchTrainer(
        train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(num_workers=settings.num_workers, use_gpu=settings.use_gpu),
        run_config=RunConfig(
            name=run_name,
            storage_path=settings.storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=2),
            failure_config=FailureConfig(max_failures=2),
        ),
    )


def default_run_name() -> str:
    return "mnist-resnet18-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")


def main() -> None:
    settings = Settings.from_env()
    ray_init_with_repo()
    settings.describe()
    run_name = os.environ.get("RUN_NAME") or default_run_name()
    print(f"run_name={run_name}  (export RUN_NAME={run_name} to resume this run later)")
    result = build_trainer(settings, run_name).fit()
    print("final metrics:", result.metrics)
    print("checkpoint:", result.checkpoint)
    print("run directory:", result.path)


if __name__ == "__main__":
    main()
