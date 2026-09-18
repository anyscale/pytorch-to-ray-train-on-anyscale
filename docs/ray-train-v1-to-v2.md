# Ray Train v1 to v2

Ray Train v2 changed how Tune wraps a training run and how a run resumes, and moved a few
context getters to a different call. This repo's scripts (`src/train_ray_train.py`,
`src/tune_ray_train.py`) are written against v2. If you have older code, or you find v1
patterns in a blog post or an older notebook, this page is the map from one to the other.

Tracking issue for the full migration: https://github.com/ray-project/ray/issues/49454

## What changed

| Area | v1 | v2 |
|---|---|---|
| Tune over Train | `ray.tune.Tuner(trainer)`: pass a `Trainer` object straight to `Tuner` as the trainable | A driver function that builds a `TorchTrainer` and calls `.fit()` itself, with `TuneReportCallback` on the trainer's `RunConfig` forwarding metrics and the checkpoint path up to Tune |
| Resuming a run | `Trainer.restore(path)` / `resume_from_checkpoint` | Rerun with the same `RunConfig(name=..., storage_path=...)`; call `ray.train.get_checkpoint()` inside the training function and load whatever state you saved |
| Run settings split | one shared `RunConfig` | `ray.train.RunConfig` for Train settings (checkpointing, failure handling), `ray.tune.RunConfig` for Tuner-level settings (the experiment name and storage path Tune itself uses) |
| Reporting | `ray.train.report(metrics, checkpoint=...)`, a barrier across all workers | Same barrier semantics, plus a `checkpoint_upload_mode` argument to control how the checkpoint upload happens |
| Context getters | `ray.train.get_context()` covered both training and trial info | Training context (`ray.train.get_context()`) still exposes `get_experiment_name`, `get_world_rank`, `get_world_size`, `get_local_rank`, `get_node_rank`, `get_storage`; trial-level getters (trial id, trial name) move to `ray.tune.get_context()`, called inside the Tune driver function |
| `ScalingConfig` | fixed `num_workers` | gains `accelerator_type`, `label_selector`, and elastic training via `num_workers=(min, max)` |

See also: [`ScalingConfig`](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html),
[`RunConfig`](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html),
[`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html),
[checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html),
[persistent storage](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html),
[hyperparameter optimization guide](https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html).

## Snippet 1: Tune over Train

v1: hand `Tuner` the `Trainer` object itself.

```python
# v1
trainer = TorchTrainer(
    train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
)
tuner = ray.tune.Tuner(
    trainer,
    param_space={"train_loop_config": {"lr": ray.tune.loguniform(1e-4, 1e-2)}},
    tune_config=ray.tune.TuneConfig(metric="loss", mode="min", num_samples=num_samples),
)
results = tuner.fit()
```

v2: a driver function builds and fits the trainer itself, and a callback carries the
metrics and checkpoint path up to Tune. This is what `src/tune_ray_train.py` does:

```python
# v2
def train_driver_fn(config: dict) -> None:
    trial_id = ray.tune.get_context().get_trial_id()
    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=config["train_loop_config"],
        scaling_config=ScalingConfig(num_workers=config["num_workers"], use_gpu=config["use_gpu"]),
        run_config=RunConfig(
            name=f"train-trial_id={trial_id}",
            storage_path=config["storage_path"],
            callbacks=[TuneReportCallback()],  # metrics + checkpoint path flow up to Tune
        ),
    )
    trainer.fit()

tuner = ray.tune.Tuner(
    train_driver_fn,
    param_space=build_param_space(settings),
    tune_config=ray.tune.TuneConfig(metric="loss", mode="min", num_samples=num_samples),
)
results = tuner.fit()
```

## Snippet 2: resuming a run

v1: restore the trainer from a checkpoint path.

```python
# v1
trainer = TorchTrainer.restore(path_to_experiment)
result = trainer.fit()
```

v2: rerun with the same `RunConfig(name, storage_path)` and read the checkpoint back inside
the training function yourself. This is what `src/train_ray_train.py` does: `RUN_NAME` set
to the same value as a previous run, plus the same `STORAGE_PATH`, is what makes it a resume
rather than a fresh run:

```python
# v2
def load_checkpoint_state(model, optimizer) -> int:
    checkpoint = ray.train.get_checkpoint()
    if not checkpoint:
        return 0
    with checkpoint.as_directory() as ckpt_dir:
        model.module.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu"))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu"))
        start_epoch = torch.load(os.path.join(ckpt_dir, "extra_state.pt"))["epoch"] + 1
    return start_epoch

# same run_name + same storage_path as the interrupted run = resume, not restart
trainer = TorchTrainer(
    train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=settings.num_workers, use_gpu=settings.use_gpu),
    run_config=RunConfig(name=run_name, storage_path=settings.storage_path),
)
result = trainer.fit()
```
