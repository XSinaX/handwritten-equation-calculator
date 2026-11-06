"""Inference for the three end-to-end image-to-CTC model pairs."""

from functools import lru_cache
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
from fastapi import HTTPException

from ml.end_to_end.ctc.decoder import greedy_decode
from ml.end_to_end.ctc.tokenizer import CTCTokenizer

CTC_MODELS = {
    "cnn-gru": "cnn-gru.pt",
    "resnet-bilstm": "resnet-bilstm.pt",
    "vit-transformer": "vision_transformer-transformer.pt",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "ml" / "trained_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = CTCTokenizer()


@lru_cache(maxsize=3)
def load_model(model_name: str):
    """Load model once, then reuse it for future requests."""
    if model_name not in CTC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

    model_path = MODEL_DIR / CTC_MODELS[model_name]
    if not model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found: {model_path.name}")

    try:
        # workaround for changing the import path of the model class when loading a checkpoint
        sys.modules.setdefault("ml.models", importlib.import_module("ml.end_to_end"))
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model = model.to(DEVICE)
        model.eval()
        return model
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        raise HTTPException(status_code=500, detail=f"Could not load {model_path.name}: {error}") from error


def recognize_ctc(processed_image: np.ndarray, model_name: str) -> str:
    """Run the model on a preprocessed uint8 grayscale equation image."""
    model = load_model(model_name)

    image_tensor = torch.from_numpy(processed_image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        return greedy_decode(logits, TOKENIZER)[0]
