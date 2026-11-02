import torch
from torch import nn

from ml.config import IMAGE_HEIGHT


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features)
        features = self.features(features)

        return self.activation(features + residual)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.expected_height = IMAGE_HEIGHT

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.output_size = 128 * (IMAGE_HEIGHT // 4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input:
            images: [batch, 1, height, width]

        Output:
            sequence: [batch, time, features]
        """
        if images.ndim != 4:
            raise ValueError("Images must have shape [batch, channels, height, width]")

        if images.shape[1] != 1:
            raise ValueError("ResNet expects grayscale images")

        if images.shape[2] != self.expected_height:
            raise ValueError(f"Expected image height {self.expected_height}, " f"received {images.shape[2]}")

        features = self.stem(images)
        features = self.features(features)

        batch_size, channels, height, width = features.shape

        features = features.permute(0, 3, 1, 2)

        sequence = features.reshape(
            batch_size,
            width,
            channels * height,
        )

        return sequence
