import math

import torch
from torch import nn

from shared.config import IMAGE_HEIGHT

PATCH_SIZE = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 4
NUM_HEADS = 4
FEEDFORWARD_SIZE = 256
DROPOUT = 0.1


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        if IMAGE_HEIGHT % PATCH_SIZE != 0:
            raise ValueError("Image height must be divisible by patch size")

        if HIDDEN_SIZE % NUM_HEADS != 0:
            raise ValueError("Hidden size must be divisible by number of heads")

        self.expected_height = IMAGE_HEIGHT
        self.patch_size = PATCH_SIZE

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=HIDDEN_SIZE,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )

        self.column_projection = nn.Linear(
            in_features=HIDDEN_SIZE * (IMAGE_HEIGHT // PATCH_SIZE),
            out_features=HIDDEN_SIZE,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_SIZE,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=NUM_LAYERS,
            norm=nn.LayerNorm(HIDDEN_SIZE),
        )

        self.output_size = HIDDEN_SIZE

    def _position_encoding(
        self,
        sequence_length: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=dtype,
        ).unsqueeze(1)

        dimensions = torch.arange(0, hidden_size, 2, device=device, dtype=dtype)
        frequencies = torch.exp(dimensions * (-math.log(10000.0) / hidden_size))

        encoding = torch.zeros(
            sequence_length,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])

        return encoding.unsqueeze(0)

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
            raise ValueError("VisionTransformer expects grayscale images")

        if images.shape[2] != self.expected_height:
            raise ValueError(f"Expected image height {self.expected_height}, " f"received {images.shape[2]}")

        if images.shape[3] % self.patch_size != 0:
            raise ValueError("Image width must be divisible by patch size")

        features = self.patch_embedding(images)

        batch_size, hidden_size, height, width = features.shape

        features = features.permute(0, 3, 2, 1)
        features = features.reshape(
            batch_size,
            width,
            height * hidden_size,
        )
        features = self.column_projection(features)

        position_encoding = self._position_encoding(
            sequence_length=width,
            hidden_size=hidden_size,
            device=features.device,
            dtype=features.dtype,
        )

        sequence = self.transformer(features + position_encoding)

        return sequence
