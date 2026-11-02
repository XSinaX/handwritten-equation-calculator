from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException

from ml.config import IMAGE_HEIGHT, SYMBOL_WIDTH
from ml.preprocessing import preprocess_symbol_image
from ml.segmentation_based.models import create_model
from ml.segmentation_based.segmentation import SEGMENTATION_METHODS

# Frontend model names -> (segmentation algorithm, symbol classifier).
SEGMENTATION_MODELS = {
    "seg-aug-cnn": ("augmented_CCL", "cnn"),
    "seg-aug-resnet": ("augmented_CCL", "resnet"),
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "ml" / "trained_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def load_classifier(classifier_name: str):
    """Load a trained symbol classifier once, including its saved token order."""
    if classifier_name not in ("cnn", "resnet"):
        raise HTTPException(status_code=400, detail=f"Unsupported classifier: {classifier_name}")

    model_path = MODEL_DIR / f"symbol-{classifier_name}.pt"
    if not model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        vocabulary = tuple(checkpoint["vocabulary"])
        expected_size = (IMAGE_HEIGHT, SYMBOL_WIDTH)
        checkpoint_size = (checkpoint["image_height"], checkpoint["image_width"])
        if checkpoint_size != expected_size:
            raise ValueError(f"Checkpoint size {checkpoint_size} does not match {expected_size}")

        model = create_model(classifier_name, num_classes=len(vocabulary))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()
        return model, vocabulary
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load {classifier_name} checkpoint: {error}",
        ) from error


def recognize_segmentation(processed_image: np.ndarray, model_name: str) -> str:
    """Segment an already-preprocessed equation, classify symbols, and join them."""
    if model_name not in SEGMENTATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
    if processed_image.ndim != 2 or processed_image.dtype != np.uint8:
        raise HTTPException(status_code=400, detail="Expected a preprocessed 2-D uint8 equation image")

    segmentation_name, classifier_name = SEGMENTATION_MODELS[model_name]
    model, vocabulary = load_classifier(classifier_name)

    segments = SEGMENTATION_METHODS[segmentation_name](processed_image)
    if not segments:
        return ""

    # Each crop is resized and centered on a 128x128 white canvas (5px padding).
    segments.sort(key=lambda segment: (segment.bbox[0], segment.bbox[1]))
    symbol_images = np.stack([preprocess_symbol_image(segment.image) for segment in segments])

    images = torch.from_numpy(1.0 - symbol_images.astype(np.float32) / 255.0)
    images = images.unsqueeze(1).to(DEVICE)

    with torch.inference_mode():
        predictions = model(images).argmax(dim=1).cpu().tolist()

    return "".join(vocabulary[index] for index in predictions)
