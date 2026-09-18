"""Stage 03: sweep hyperparameters of the distributed run with Ray Tune.

    python src/tune_ray_train.py
    NUM_SAMPLES=4 MAX_CONCURRENT_TRIALS=2 NUM_WORKERS=2 python src/tune_ray_train.py

The Ray Train v2 pattern: each Tune trial runs a small *driver function* that
builds a TorchTrainer and calls fit(). TuneReportCallback forwards the worker
metrics (and the checkpoint path) up to Tune.
Docs: https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html
"""
from __future__ import annotations

import datetime
import os
import sys

import pyarrow.fs
import torch

import ray
import ray.train
import ray.tune
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.integration.ray_train import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.settings import Settings, ray_init_with_repo  # noqa: E402
from src.train_ray_train import train_loop_per_worker  # noqa: E402


def train_driver_fn(config: dict) -> None:
    """Runs once per Tune trial, on 1 CPU. Launches the distributed Train run and waits for it."""
    trial_id = ray.tune.get_context().get_trial_id()
    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=config["train_loop_config"],
        scaling_config=ScalingConfig(num_workers=config["num_workers"], use_gpu=config["use_gpu"]),
        run_config=RunConfig(
            name=f"train-trial_id={trial_id}",  # stable name, so a restarted driver resumes in place
            storage_path=config["storage_path"],
            callbacks=[TuneReportCallback()],  # metrics + checkpoint path flow up to Tune
            # Scored on training loss: fine here since this repo has no validation split.
            # A real workload should report a validation metric and score on that instead.
            checkpoint_config=CheckpointConfig(
                num_to_keep=1, checkpoint_score_attribute="loss", checkpoint_score_order="min"
            ),
            failure_config=FailureConfig(max_failures=2),  # worker-level retries
        ),
    )
    trainer.fit()


def build_param_space(settings: Settings) -> dict:
    train_loop_config = settings.as_train_loop_config()
    train_loop_config["lr"] = ray.tune.loguniform(1e-4, 1e-2)  # the only sampled parts
    train_loop_config["global_batch_size"] = ray.tune.choice([64, 128, 256])
    return {
        "num_workers": settings.num_workers,
        "use_gpu": settings.use_gpu,
        "storage_path": settings.storage_path,
        "train_loop_config": train_loop_config,
    }


def build_tuner(settings: Settings, experiment_name: str, num_samples: int, max_concurrent_trials: int) -> ray.tune.Tuner:
    return ray.tune.Tuner(
        train_driver_fn,
        param_space=build_param_space(settings),
        tune_config=ray.tune.TuneConfig(
            # Pruned on training loss: fine here since this repo has no validation split.
            # A real workload should report a validation metric and prune on that instead.
            metric="loss",
            mode="min",
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,  # peak GPUs = this x num_workers
            scheduler=ASHAScheduler(
                time_attr="training_iteration",  # one Train report = one iteration
                grace_period=1,
                max_t=settings.num_epochs,
                reduction_factor=2,
            ),
        ),
        run_config=ray.tune.RunConfig(
            name=experiment_name,
            storage_path=settings.storage_path,
            failure_config=ray.tune.FailureConfig(max_failures=1),  # driver-level retries
        ),
    )


def best_checkpoint(results: ray.tune.ResultGrid, storage_path: str) -> Checkpoint:
    """TuneReportCallback attaches the Train checkpoint path as the metric `checkpoint_path`.

    That path is scheme-less on cloud storage: Ray strips the URI scheme
    (e.g. "s3://") before handing it to Tune, because the scheme belongs on
    Checkpoint.filesystem, not baked into the path string. `Checkpoint(path=...)`
    with no filesystem asks pyarrow to infer one from the path itself, and on
    a scheme-less cloud path that fails with "URI has empty scheme". It never
    shows up on local storage, where paths need no scheme either way, which is
    why a local smoke test can't catch it. The fix is to hand Checkpoint the
    filesystem explicitly, derived from the run's own storage_path (which DOES
    still have its scheme), instead of trying to recover it from the path.
    """
    best = results.get_best_result()
    path = best.metrics["checkpoint_path"]
    filesystem, _ = pyarrow.fs.FileSystem.from_uri(storage_path)
    return Checkpoint(path=path, filesystem=filesystem)


def main() -> None:
    settings = Settings.from_env()
    ray_init_with_repo()
    settings.describe()
    num_samples = int(os.environ.get("NUM_SAMPLES", "4"))
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_TRIALS", "2"))
    name = os.environ.get("EXPERIMENT_NAME") or "tune-mnist-" + datetime.datetime.now(datetime.UTC).strftime(
        "%Y%m%d-%H%M%S"
    )
    print(
        f"experiment={name} trials={num_samples} concurrent={max_concurrent} "
        f"peak_workers={max_concurrent * settings.num_workers}"
    )
    results = build_tuner(settings, name, num_samples, max_concurrent).fit()
    df = results.get_dataframe()
    cols = [
        c
        for c in ["trial_id", "loss", "epoch", "config/train_loop_config/lr", "config/train_loop_config/global_batch_size"]
        if c in df.columns
    ]
    print(df[cols].sort_values("loss").to_string(index=False))
    best = results.get_best_result()
    print("best hyperparameters:", {k: best.config["train_loop_config"][k] for k in ("lr", "global_batch_size")})
    checkpoint = best_checkpoint(results, settings.storage_path)
    print("best checkpoint:", checkpoint)

    # Constructing a Checkpoint proves nothing by itself: prove it actually reads back.
    with checkpoint.as_directory() as ckpt_dir:
        state_dict = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    num_params = sum(p.numel() for p in state_dict.values())
    print(f"checkpoint loads: {len(state_dict)} tensors, {num_params:,} parameters")


if __name__ == "__main__":
    main()
