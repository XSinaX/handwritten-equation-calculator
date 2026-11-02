import torch
from torch import nn
from ml.config import IMAGE_HEIGHT


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.expected_height = IMAGE_HEIGHT

        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # layer 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # layer 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output_size = 64 * (IMAGE_HEIGHT // 4)

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
            raise ValueError(f"Expected image height {self.expected_height}, " f"received {images.shape[2]}")

        features = self.features(images)

        batch_size, channels, height, width = features.shape

        features = features.permute(0, 3, 1, 2)

        sequence = features.reshape(
            batch_size,
            width,
            channels * height,
        )

        return sequence
