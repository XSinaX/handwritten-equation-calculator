import math

import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be divisible by number of heads")

        self.input_projection = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        self.output_size = hidden_size

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
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )

        return encoding.unsqueeze(0)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Input:
            features: [batch, time, image_features]
            padding_mask: [batch, time], where True marks padding

        Output:
            sequence_output: [batch, time, hidden_size]
        """
        if features.ndim != 3:
            raise ValueError("Features must have shape [batch, time, features]")

        sequence_output = self.input_projection(features)
        sequence_output = sequence_output + self._position_encoding(
            sequence_length=sequence_output.shape[1],
            hidden_size=sequence_output.shape[2],
            device=sequence_output.device,
            dtype=sequence_output.dtype,
        )

        return self.transformer(
            sequence_output,
            src_key_padding_mask=padding_mask,
        )

