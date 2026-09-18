# Running this repo on a Kubernetes cloud

Everything in this repo also runs on an Anyscale cloud backed by Kubernetes (EKS, GKE, AKS).
The scripts and env vars are unchanged. What changes is where instance type names come from,
how much GPU capacity you have to work with, and, on some clouds, where storage lives.

## Where instance type names come from

On a VM cloud (the AWS example this repo ships with), `instance_type` is a real cloud VM type
like `g6.xlarge`. On a Kubernetes cloud it is not a VM type at all, but a pod size (a CPU,
memory, and optional GPU count and type) that your platform team defines in the Anyscale
operator's Helm values when the cloud was set up.

You cannot guess these names. Get them one of two ways:

- Ask your platform team for the pod size names available on your cloud.
- Read one off a compute config that's already registered: `anyscale compute-config get --name <an-existing-config>`.

`jobs/compute_config.kubernetes.example.yaml` in this repo is the template: copy it, fill in
your cloud name and pod sizes, and register it the same way as the AWS example. See
[compute on Kubernetes](https://docs.anyscale.com/configuration/compute/kubernetes) and
[Kubernetes clouds](https://docs.anyscale.com/clouds/kubernetes).

## No GPU on the head node, still

The rule this repo follows on every cloud applies here too: the head node has no GPU. All
three job configs (`jobs/job_01_pytorch.yaml`, `job_02_ray_train.yaml`, `job_03_tune.yaml`)
assume this. It's worth restating one consequence that's easy to miss on Kubernetes: Ray
Tune's trial driver processes (the small, CPU-only function each trial runs to build and fit
a `TorchTrainer`, see `docs/ray-train-v1-to-v2.md`) land on the head node. That is deliberate,
not a misconfiguration. Those drivers are cheap, and keeping them off the GPU worker pods
means a preempted or rescaled GPU pod doesn't take a trial driver down with it.

## GPU caps and sizing your run against them

Kubernetes clouds are commonly set up with a hard cap on GPUs available per workspace or
project (a node pool sized by your platform team, sometimes a `ResourceQuota`). Before you
raise `NUM_WORKERS` or `MAX_CONCURRENT_TRIALS`, know that cap:

- `src/train_ray_train.py` uses `NUM_WORKERS` GPUs at once.
- `src/tune_ray_train.py` uses up to `MAX_CONCURRENT_TRIALS x NUM_WORKERS` GPUs at once:
  that's the number to check against your cap, not `NUM_WORKERS` alone.

If that product exceeds what your node pool can provide, the run does not fail outright; see
"node pools" below for what it looks like instead.

## Storage

Two assumptions this repo makes on a VM cloud don't always hold on Kubernetes:

- `/mnt/cluster_storage` (what `src/settings.py` checks first) may not be mounted on every
  Kubernetes cloud. When it's absent, `resolve_storage_path()` falls through to
  `$ANYSCALE_ARTIFACT_STORAGE`.
- `$ANYSCALE_ARTIFACT_STORAGE` itself may point at Azure Blob Storage rather than S3 or GCS,
  depending on the cloud. Reading and writing it then needs `adlfs` installed alongside the
  rest of this repo's requirements.

If either of those causes trouble, or you just want to be explicit about where checkpoints
go, set `STORAGE_PATH="artifact://pytorch-to-ray-train"` (or a `jobs/`-suffixed variant, as
the shipped job configs do): it always resolves against whatever
`$ANYSCALE_ARTIFACT_STORAGE` is configured to on your cloud, so the same env var override
works regardless of the backing object store. See
[storage](https://docs.anyscale.com/storage),
[shared storage](https://docs.anyscale.com/storage/shared), and
[local storage](https://docs.anyscale.com/storage/local).

## Images

The image this repo builds (`anyscale/image/pytorch-to-ray-train:1`, from `./containerfile`)
is not automatically shared across organizations or clouds. On a new Kubernetes cloud you
have two options:

- Build your own copy from the same `containerfile` for your organization:
  `anyscale image build -n pytorch-to-ray-train -f containerfile --ray-version 2.58.0`.
- Register an existing image your platform team already built or pushed to a registry your
  cluster can pull from.

See [build image](https://docs.anyscale.com/container-image/build-image),
[custom image](https://docs.anyscale.com/container-image/custom-image), and
[base images](https://docs.anyscale.com/reference/base-images).

## Node pools and a run that waits forever

On Kubernetes, GPU worker pods are scheduled onto a specific node pool (or pools) your
platform team configured for the cloud. If a job asks for more GPU workers than that pool can
provide, or asks for a pod size the pool doesn't have at all, the job does not error out. It
sits with pending pods, waiting for capacity that will never arrive. If a job you submitted
seems stuck with the cluster up but no training output, check with your platform team whether
the node pool for that pod size has room, before assuming the script itself is broken.
