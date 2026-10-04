import math
import torch
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_height: int,
        patch_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        feedforward_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if image_height % patch_size != 0:
            raise ValueError("Image height must be divisible by patch size")

        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be divisible by number of heads")

        self.expected_height = image_height
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.column_projection = nn.Linear(
            in_features=hidden_size * (image_height // patch_size),
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
            raise ValueError(
                f"Expected image height {self.expected_height}, "
                f"received {images.shape[2]}"
            )

        if images.shape[3] % self.patch_size != 0:
            raise ValueError("Image width must be divisible by patch size")

        features = self.patch_embedding(images)
        batch_size, hidden_size, height, width = features.shape

        # Combine each vertical patch column into one token so the transformer attends from left to right and remains compatible with the CTC head.
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

        return self.transformer(features + position_encoding)

