from pathlib import Path

import torch
from fastapi import HTTPException

MODEL_FOLDER = Path(__file__).resolve().parents[2] / "models"


def load_model(
    image_model: str,
    sequence_model: str,
):
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

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model '{model_name}': {error}",
        )
