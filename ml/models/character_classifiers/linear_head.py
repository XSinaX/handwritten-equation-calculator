import torch
from torch import nn


class LinearClassifierHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.classifier = nn.Linear(
            in_features=input_size,
            out_features=num_classes,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            features: [batch, time, sequence_features]

        Output:
            logits: [batch, time, character_classes]
        """
        return self.classifier(features)
