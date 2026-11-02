import math

import torch
from torch import nn

from ml.config import NUM_CLASSES

HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FEEDFORWARD_SIZE = 256
DROPOUT = 0.1


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()

        if HIDDEN_SIZE % NUM_HEADS != 0:
            raise ValueError("Hidden size must be divisible by number of heads")

        self.input_projection = nn.Linear(
            in_features=input_size,
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

        # Predict a class at every time step
        self.classifier = nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=NUM_CLASSES,
        )

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: [batch, time, image_features]

        Output:
            logits: [batch, time, num_classes]
        """
        if features.ndim != 3:
            raise ValueError("Features must have shape [batch, time, features]")

        sequence_output = self.input_projection(features)

        position_encoding = self._position_encoding(
            sequence_length=sequence_output.shape[1],
            hidden_size=sequence_output.shape[2],
            device=sequence_output.device,
            dtype=sequence_output.dtype,
        )

        sequence_output = self.transformer(sequence_output + position_encoding)
        logits = self.classifier(sequence_output)

        return logits
