import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )

        # Forward and backward outputs are concatenated.
        self.output_size = hidden_size * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: [batch, time, CNN_features]

        Output:
            sequence_output: [batch, time, hidden_size * 2]
        """
        sequence_output, _ = self.lstm(features)
        return sequence_output
