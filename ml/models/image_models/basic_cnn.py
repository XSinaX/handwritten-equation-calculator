import torch
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, image_height: int):
        super().__init__()

        self.expected_height = image_height

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output_size = 64 * (image_height // 4)

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
            raise ValueError("BasicCNN expects grayscale images")

        if images.shape[2] != self.expected_height:
            raise ValueError(
                f"Expected image height {self.expected_height}, "
                f"received {images.shape[2]}"
            )

        features = self.features(images)

        batch_size, channels, height, width = features.shape

        features = features.permute(0, 3, 1, 2)

        sequence = features.reshape(
            batch_size,
            width,
            channels * height,
        )

        return sequence
