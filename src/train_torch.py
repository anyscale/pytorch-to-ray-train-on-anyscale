"""Stage 01: the plain PyTorch training loop as a script, no Ray involved.

    python src/train_torch.py                 # 2 epochs on a GPU, wherever the GPU is
    SMOKE_TEST=1 python src/train_torch.py    # 2,048 samples, 1 epoch, quick check
    USE_GPU=0 python src/train_torch.py       # force CPU, no Ray involved
"""
from __future__ import annotations

import csv
import datetime
import os
import sys

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_data_loader  # noqa: E402
from src.model import build_resnet18  # noqa: E402
from src.settings import Settings  # noqa: E402


def pick_device(use_gpu: bool) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if use_gpu:
        print("USE_GPU=1 but no CUDA device is visible from this process. Falling back to CPU.")
    return "cpu"


def train_one_epoch(model, data_loader, criterion, optimizer, device) -> float:
    model.train()
    loss = torch.tensor(0.0)
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)  # you move every batch yourself
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def save_checkpoint_and_metrics(model, metrics: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.csv"), "a", newline="") as f:
        csv.writer(f).writerow([metrics["epoch"], metrics["loss"]])
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


def train_loop_torch(settings: Settings, output_dir: str, device: str) -> dict:
    model = build_resnet18().to(device)  # you place the model yourself
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=settings.lr)
    data_loader = build_data_loader(settings.data_root, settings.global_batch_size, settings.subset_size)
    metrics: dict = {}
    for epoch in range(settings.num_epochs):
        loss = train_one_epoch(model, data_loader, criterion, optimizer, device)
        metrics = {"epoch": epoch, "loss": loss}
        print(metrics)
        save_checkpoint_and_metrics(model, metrics, output_dir)
    return metrics


def main() -> None:
    """Run the loop where a GPU actually is.

    Started on a machine with a GPU (a laptop, a GPU workstation), this just
    trains. Started on an Anyscale head node, which has no GPU by design, it
    asks Ray for one GPU worker and runs the SAME unchanged function there.
    That placement is the first thing Ray does for you, before any Ray Train.
    """
    settings = Settings.from_env()
    settings.describe()
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(settings.storage_path, "01_pytorch", stamp)

    if not settings.use_gpu or torch.cuda.is_available():
        device = pick_device(settings.use_gpu)
        print(f"training here: device={device} output_dir={output_dir}")
        final = train_loop_torch(settings, output_dir, device)
    else:
        import ray

        if not ray.is_initialized():
            ray.init()
        print(f"no GPU on this node; placing the same function on a GPU worker. output_dir={output_dir}")

        @ray.remote(num_gpus=1)
        def train_on_gpu_worker() -> dict:
            import socket

            print(f"training on {socket.gethostname()} / {torch.cuda.get_device_name(0)}")
            return train_loop_torch(settings, output_dir, "cuda")

        final = ray.get(train_on_gpu_worker.remote(), timeout=3600)

    print(f"done: {final}. Checkpoint: {output_dir}/model.pt")


if __name__ == "__main__":
    main()
