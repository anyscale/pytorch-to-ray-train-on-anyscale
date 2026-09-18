# PyTorch to Ray Train on Anyscale

A hands-on path for an engineer who has PyTorch training code and has never used Ray or
Anyscale. Five notebooks take one training loop from a single GPU to distributed training
with Ray Train, a hyperparameter sweep with Ray Tune, and unattended Anyscale Jobs. Every
notebook links the public docs, templates, and courses that go deeper.

The running example is MNIST with ResNet18 because it trains in about a minute per epoch on
one GPU and matches the official Ray Train PyTorch guide. Swap in your own model and data
once the shape is familiar; the section "Bring your own training script" below says how.

## How to use this repo

1. Open `notebooks/00_before_you_start.ipynb` in your browser on GitHub and follow it to get
   a workspace running with this repo in it, using both the fast and the durable way to bring
   in dependencies.
2. In the workspace, open Jupyter (or VS Code) and run notebooks 01 through 03 in order. Each
   one runs top to bottom in one go.
3. Open notebook 04 and submit the `src/` scripts as Anyscale Jobs, first from the workspace
   and then from your laptop.

Every notebook and script reads its configuration from environment variables with defaults
that fit a workspace with GPU workers. The full list, with defaults, is in "Configuration"
below and at the top of `src/settings.py`.

## The journey

| Notebook | You learn | Minutes | GPUs |
|---|---|---|---|
| `00_before_you_start` | Access, creating a workspace, getting code in, dependencies the fast way and the durable way, verifying GPUs and storage | 20 | 2 |
| `01_pytorch_as_is` | Your unchanged PyTorch loop, reaching a GPU worker from a GPU-less head | 15 | 1 |
| `02_ray_train_distributed` | Converting to Ray Train v2: prepare model and data, report checkpoints, scale across nodes, recover from a worker failure, resume a run | 20 | 2 to 4 |
| `03_ray_tune_with_ray_train` | Sweeping hyperparameters over the distributed trainer: driver functions, resource math, early stopping, loading the best checkpoint | 20 | 4 |
| `04_run_as_anyscale_jobs` | Submitting the scripts as jobs from a workspace and a laptop, durable outputs, and triaging a job that never got a cluster | 30 | 2 to 4 |

This repo does not ship separate notebooks for Ray Data ingestion or experiment tracking.
Both were considered and built once, then cut: they are one-page variants of
`src/train_ray_train.py` (swap the `DataLoader` for a Ray Dataset, or add a few lines of
MLflow or TensorBoard logging inside `train_loop_per_worker`), and hand-building and
validating them here would cost more than they teach. Ask the Anyscale `ray-train` agent
skill to generate either one against this repo's code when you need it.

## Configuration

Every notebook and script builds its settings from `src.settings.Settings.from_env()`. The
table below is a summary; `src/settings.py` is the source of truth, including how
`STORAGE_PATH` is resolved when it is not set explicitly.

| Variable | Meaning | Default |
|---|---|---|
| `NUM_WORKERS` | Ray Train workers | `2` |
| `USE_GPU` | Train on GPU (`1`/`true`/`yes`) or CPU (`0`/`false`/`no`) | `1` |
| `NUM_EPOCHS` | Training epochs | `2` (forced to `1` when `SMOKE_TEST=1`) |
| `GLOBAL_BATCH_SIZE` | Batch size, split evenly across workers | `128` |
| `LR` | Learning rate | `1e-3` |
| `SMOKE_TEST` | `1` trains on a 2,048-sample subset for 1 epoch | `0` |
| `DATA_ROOT` | Per-node torchvision download root | `/mnt/local_storage/data` if that mount exists, else `/tmp/data` |
| `STORAGE_PATH` | Ray Train `RunConfig.storage_path`; accepts an `artifact://<subpath>` sentinel that resolves against `ANYSCALE_ARTIFACT_STORAGE` | shared cluster storage if mounted, else artifact storage, else `/tmp/pytorch-to-ray-train` with a warning |
| `MLFLOW_TRACKING_URI` | Tracking URI for MLflow logging you add yourself | a file store derived from `STORAGE_PATH` |

`src/train_ray_train.py` and `src/tune_ray_train.py` read a few more variables of their own
(`RUN_NAME`, `NUM_SAMPLES`, `MAX_CONCURRENT_TRIALS`, `EXPERIMENT_NAME`,
`RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S`), documented in each script's own docstring and in
the job configs under `jobs/` that set them.

## Repository layout

```
notebooks/       the five notebooks in the journey above, run top to bottom in a workspace
src/              settings.py (env-driven config), model.py, data.py, and the three stage
                  scripts: train_torch.py, train_ray_train.py, tune_ray_train.py
jobs/             job configs (job_01_pytorch.yaml, job_02_ray_train.yaml, job_03_tune.yaml),
                  the compute config this repo was validated on, a Kubernetes compute config
                  template, and a workspace example
tests/            unit tests (test_settings.py, test_model_data.py, test_tune_ray_train.py),
                  job and notebook runners (run_jobs.sh, run_notebooks.sh), and check_links.py
docs/             ray-train-v1-to-v2.md (what changed between Ray Train API versions) and
                  kubernetes-clouds.md (running this repo on a Kubernetes-backed cloud)
containerfile     the image every notebook and job runs on: anyscale/ray:2.58.0-py312-cu128
                  plus the pins in requirements.txt
requirements.txt  torch, torchvision, tensorboard pins, kept identical to containerfile
```

The compute config used to validate this repo, `jobs/compute_config.aws.yaml`, registers as
`gpu-multinode-dev`: an `m5.xlarge` head with no GPU, plus three interchangeable single-GPU
worker groups (T4, A10G, L4), each scaling from zero. Listing several GPU types buys capacity
fallback, not cost preference: Anyscale's autoscaler picks whichever type it can get, and
worker group order is not honored as a priority. This was observed directly during
development, when a job landed on the A10G worker group even though the cheaper T4 was listed
first, because GPU capacity had run out in three separate availability zones. If your
platform team's cloud is short on one GPU type, listing a few alternatives like this is worth
doing for the same reason.

## Bring your own training script

The scripts in `src/` are the notebooks' code collected into files, and they are meant to be
copied.

1. Start from `src/train_torch.py`. Replace `build_resnet18` (in `src/model.py`) and
   `build_data_loader` (in `src/data.py`) with your model and loader, and check the loop
   runs on one GPU (`notebooks/01_pytorch_as_is`).
2. Move to `src/train_ray_train.py`. Put your loop inside `train_loop_per_worker`, wrap the
   model with `ray.train.torch.prepare_model`, the loader with
   `ray.train.torch.prepare_data_loader`, report metrics and a rank-0 checkpoint with
   `report_checkpoint` (which calls `ray.train.report`), and call `load_checkpoint_state` at
   the start so runs resume from `ray.train.get_checkpoint()`
   (`notebooks/02_ray_train_distributed`).
3. Keep `src/tune_ray_train.py` almost unchanged: `train_driver_fn` already builds and fits a
   `TorchTrainer` around whatever `train_loop_per_worker` you wrote in step 2, so it only
   needs the names of the hyperparameters you want to sweep, added to `build_param_space`
   (`notebooks/03_ray_tune_with_ray_train`).
4. Copy `jobs/job_02_ray_train.yaml`, change `compute_config` to a name registered on your
   cloud, and submit it with `anyscale job submit -f <your-copy>.yaml --working-dir .`
   (`notebooks/04_run_as_anyscale_jobs`).

Things that usually need attention when the model is real: the per-worker batch size and
learning rate implied by `GLOBAL_BATCH_SIZE // world_size`, data that must not be downloaded
fresh on every node (`src/data.py`'s file lock is the minimum version of this), and
checkpoints that need optimizer and scheduler state, which `train_ray_train.py` already
saves and restores.

## Running on a Kubernetes-backed cloud

Nothing in the notebooks or scripts names a cloud VM type directly. Read
`docs/kubernetes-clouds.md` for where instance type names come from on a Kubernetes cloud,
how a GPU cap should change `NUM_WORKERS` and `MAX_CONCURRENT_TRIALS`, and what a job looks
like when it never gets a cluster because the capacity it asked for isn't there.

## What was validated, and how

Every script in `src/` was proven as an Anyscale Job, on its own cluster, before any notebook
was written around it:

- `jobs/job_01_pytorch.yaml` (`src/train_torch.py`): plain PyTorch reaching a GPU worker from
  a GPU-less head node, with a checkpoint written out.
- `jobs/job_02_ray_train.yaml` (`src/train_ray_train.py`): Ray Train v2 running across two
  GPU worker nodes, with checkpoints landing in Anyscale artifact storage so they outlive the
  job's cluster.
- `jobs/job_03_tune.yaml` (`src/tune_ray_train.py`): a four-trial Ray Tune sweep
  (`NUM_SAMPLES=4`, `MAX_CONCURRENT_TRIALS=2`) over the distributed trainer, with an ASHA
  scheduler stopping weak trials early.

To repeat any of this yourself:

```bash
bash tests/run_jobs.sh              # submit all three job configs and wait for SUCCEEDED
bash tests/run_notebooks.sh full        # every notebook, in place
bash tests/run_notebooks.sh smoke-cpu   # notebooks 01 to 03 on CPU with a data subset
python tests/check_links.py             # every URL in the README, docs, and notebooks answers
```

The unit tests are separate from the above and need no cluster:

```bash
uv run --with pytest --with torch --with torchvision --with filelock --with pydantic python -m pytest tests/ -q
```

That currently passes 20 tests. `pydantic` is needed because one test imports Ray Train code
at module level.

## Public resources

| | |
|---|---|
| Ray Train | [overview](https://docs.ray.io/en/latest/train/overview.html), [PyTorch guide](https://docs.ray.io/en/latest/train/getting-started-pytorch.html), [checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html), [fault tolerance](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html), [persistent storage](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html), [hyperparameter optimization](https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html), [v2 migration guide](https://github.com/ray-project/ray/issues/49454) |
| Ray Tune | [key concepts](https://docs.ray.io/en/latest/tune/key-concepts.html), [search spaces](https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html), [schedulers](https://docs.ray.io/en/latest/tune/api/schedulers.html) |
| Ray Data | [overview](https://docs.ray.io/en/latest/data/data.html), [data loading for Train](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html) |
| Anyscale | [get started](https://docs.anyscale.com/get-started), [workspaces](https://docs.anyscale.com/workspaces), [dependency management](https://docs.anyscale.com/dependency-management), [jobs](https://docs.anyscale.com/jobs), [compute configs](https://docs.anyscale.com/configuration/compute), [storage](https://docs.anyscale.com/storage), [Ray Train runtime](https://docs.anyscale.com/runtime/train), [CLI](https://docs.anyscale.com/reference/cli) |
| Templates | [workspace intro](https://github.com/anyscale/templates/tree/main/templates/workspace-intro), [job intro](https://github.com/anyscale/templates/tree/main/templates/job-intro), [distributing PyTorch](https://github.com/anyscale/templates/tree/main/templates/distributing-pytorch), [Tune + Train integration](https://github.com/anyscale/templates/tree/main/templates/ray-tune-train-integration), [all templates](https://console.anyscale.com/templates) |
| Courses | [Ray Train specialization](https://anyscale.coursifai.com/learning-paths/ray-train-specialization) |

## License

Apache-2.0. See `LICENSE`.

© 2026, Anyscale. All Rights Reserved
