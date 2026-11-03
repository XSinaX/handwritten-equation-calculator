import logging
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import HTTPException

logger = logging.getLogger(__name__)

MODEL_FOLDER = Path(__file__).resolve().parents[2] / "models"

# Only these names are accepted, so user input can never point outside
# MODEL_FOLDER (torch.load with weights_only=False can run arbitrary code).
IMAGE_MODELS = ("cnn", "resnet", "vision_transformer")
SEQUENCE_MODELS = ("gru", "bilstm", "transformer")


def validate_model_names(image_model: str, sequence_model: str) -> None:
    if image_model not in IMAGE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown image model '{image_model}'. Choose one of: {', '.join(IMAGE_MODELS)}.",
        )

    if sequence_model not in SEQUENCE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sequence model '{sequence_model}'. Choose one of: {', '.join(SEQUENCE_MODELS)}.",
        )


# Big enough to hold every combination, so switching models in the UI
# never forces a reload from disk.
@lru_cache(maxsize=len(IMAGE_MODELS) * len(SEQUENCE_MODELS))
def load_model(
    image_model: str,
    sequence_model: str,
):
    validate_model_names(image_model, sequence_model)

    model_name = f"{image_model}-{sequence_model}.pt"
    model_path = MODEL_FOLDER / model_name

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' was not found.",
        )

    try:
        model = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        model.eval()

        return model

    except Exception:
        # Full details go to the server log, not to the user
        logger.exception("Failed to load model %s", model_name)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model '{model_name}'.",
        )


def preload_all_models() -> None:
    """Load every model once at startup so the first request isn't slow."""

    for image_model in IMAGE_MODELS:
        for sequence_model in SEQUENCE_MODELS:
            try:
                load_model(image_model, sequence_model)
                logger.info("Loaded model %s-%s", image_model, sequence_model)
            except HTTPException as error:
                logger.warning("Skipped %s-%s: %s", image_model, sequence_model, error.detail)


def loaded_model_count() -> int:
    return load_model.cache_info().currsize
