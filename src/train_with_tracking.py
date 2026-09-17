"""Stage: the Ray Train loop with MLflow and TensorBoard experiment tracking.

    python src/train_with_tracking.py
    USE_GPU=0 NUM_WORKERS=2 SMOKE_TEST=1 python src/train_with_tracking.py

Docs: https://docs.ray.io/en/latest/train/user-guides/experiment-tracking.html
"""
from __future__ import annotations

import os
import sys

import mlflow
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import ray
import ray.train
import ray.train.torch
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_data_loader  # noqa: E402
from src.model import build_resnet18  # noqa: E402
from src.settings import Settings, ray_init_with_repo  # noqa: E402
from src.train_ray_train import (  # noqa: E402
    default_run_name,
    load_checkpoint_state,
    maybe_fail_once,
    report_checkpoint,
)


def train_loop_with_tracking(config: dict) -> None:
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

    run_name = ctx.get_experiment_name()
    writer = None
    tracking_enabled = False
    if rank == 0:
        try:
            mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
            mlflow.set_experiment("pytorch-to-ray-train")
            writer = SummaryWriter(os.path.join(config["tensorboard_dir"], run_name))
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(
                {
                    "lr": config["lr"],
                    "global_batch_size": config["global_batch_size"],
                    "num_epochs": config["num_epochs"],
                }
            )
            tracking_enabled = True
        except Exception as exc:  # training is the product; tracking is only the observation
            print(f"WARNING: experiment tracking unavailable, continuing without it: {exc}")

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
            if tracking_enabled:
                mlflow.log_metrics({"loss": metrics["loss"]}, step=epoch)
                writer.add_scalar("loss", metrics["loss"], epoch)
        report_checkpoint(model, optimizer, metrics, epoch)

    if rank == 0 and tracking_enabled:
        writer.close()
        mlflow.end_run()


def main() -> None:
    settings = Settings.from_env()
    if not ray.is_initialized():
        ray_init_with_repo()
    settings.describe()
    run_name = os.environ.get("RUN_NAME") or default_run_name()
    print(f"run_name={run_name}  (export RUN_NAME={run_name} to resume this run later)")

    train_loop_config = settings.as_train_loop_config()
    train_loop_config["mlflow_tracking_uri"] = settings.mlflow_tracking_uri
    train_loop_config["tensorboard_dir"] = f"{settings.storage_path}/tensorboard"

    trainer = TorchTrainer(
        train_loop_with_tracking,
        train_loop_config=train_loop_config,
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

    mlflow.set_tracking_uri(train_loop_config["mlflow_tracking_uri"])
    runs = mlflow.search_runs(experiment_names=["pytorch-to-ray-train"])
    print(f"mlflow runs recorded: {len(runs)}")

    tb_dir = os.path.join(train_loop_config["tensorboard_dir"], run_name)
    acc = EventAccumulator(tb_dir)
    acc.Reload()
    scalars = acc.Scalars("loss") if "loss" in acc.Tags().get("scalars", []) else []
    print(f"tensorboard scalars for {run_name}: {[(s.step, s.value) for s in scalars]}")


if __name__ == "__main__":
    main()
