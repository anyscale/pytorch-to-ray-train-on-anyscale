"""ResNet18 adapted to 1-channel 28x28 MNIST digits."""
import torch
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 10) -> torch.nn.Module:
    model = resnet18(num_classes=num_classes)
    # torchvision's first conv expects 3 RGB channels. MNIST is grayscale.
    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    return model
