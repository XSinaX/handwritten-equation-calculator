import torch
from torch import nn
from shared.config import NUM_CLASSES

HIDDEN_SIZE = 128
NUM_LAYERS = 2


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )

        # Forward and backward LSTM outputs are concatenated
        lstm_output_size = HIDDEN_SIZE * 2

        # Predict a class at every time step
        self.classifier = nn.Linear(
            in_features=lstm_output_size,
            out_features=NUM_CLASSES,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: [batch, time, CNN_features]

        Output:
            logits: [batch, time, num_classes]
        """

        sequence_output, _ = self.lstm(features)
        logits = self.classifier(sequence_output)

        return logits
