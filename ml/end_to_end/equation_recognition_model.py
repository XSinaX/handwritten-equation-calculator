from torch import nn

from ml.models.image_models.basic_cnn import BasicCNN
from ml.models.image_models.resnet import ResNet
from ml.models.image_models.vision_transformer import VisionTransformer
from ml.models.sequence_models.bilstm import BiLSTM
from ml.models.sequence_models.gru import GRU
from ml.models.sequence_models.transformer import Transformer


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
        elif image_model == "resnet":
            self.image_model = ResNet()
        elif image_model == "vision_transformer":
            self.image_model = VisionTransformer()
        else:
            raise ValueError(f"Unsupported image model: {image_model}")

        # Sequence model + classifier
        if sequence_model == "bilstm":
            self.sequence_model = BiLSTM(
                input_size=self.image_model.output_size,
            )
        elif sequence_model == "gru":
            self.sequence_model = GRU(
                input_size=self.image_model.output_size,
            )
        elif sequence_model == "transformer":
            self.sequence_model = Transformer(
                input_size=self.image_model.output_size,
            )
        else:
            raise ValueError(f"Unsupported sequence model: {sequence_model}")

    def forward(self, images):
        features = self.image_model(images)

        logits = self.sequence_model(features)

        return logits
