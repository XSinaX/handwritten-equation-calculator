import torch
from torch import nn


class EquationCTCLoss(nn.Module):
    def __init__(self, blank_index=0):
        super().__init__()

        self.loss_function = nn.CTCLoss(
            blank=blank_index,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        logits,
        targets,
        target_lengths,
    ):
        """
        Parameters
        ----------
        logits:
            [batch, time, classes]

        targets:
            All target sequences concatenated into one tensor.

        target_lengths:
            Length of each target sequence.

        Returns
        -------
        CTC loss.
        """

        if logits.ndim != 3:
            raise ValueError("Logits must have shape " "[batch, time, classes]")

        batch_size, time_steps, _ = logits.shape

        # CTC expects log probabilities rather than raw logits.
        log_probs = logits.log_softmax(dim=2)
        log_probs = log_probs.permute(1, 0, 2)

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=time_steps,
            dtype=torch.long,
        )

        return self.loss_function(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )
