from torch import nn

from ml.models.image_models.basic_cnn import BasicCNN
from ml.models.sequence_models.bilstm import BiLSTM


class EquationRecognitionModel(nn.Module):
    def __init__(
        self,
        image_model,
        sequence_model,
    ):
        super().__init__()

        # Image model
        if image_model == "cnn":
            self.image_model = BasicCNN()
        # TODO: Add support for other image models
        # elif image_model == "resnet":
        # elif image_model == "vision_transformer":
        else:
            raise ValueError(f"Unsupported image model: {image_model}")

        # Sequence model + classifier
        if sequence_model == "bilstm":
            self.sequence_model = BiLSTM(
                input_size=self.image_model.output_size,
            )
        # TODO: Add support for other sequence models
        # elif sequence_model == "gru":
        # elif sequence_model == "transformer":
        else:
            raise ValueError(f"Unsupported sequence model: {sequence_model}")

    def forward(self, images):
        features = self.image_model(images)

        logits = self.sequence_model(features)

        return logits
