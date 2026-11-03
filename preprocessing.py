import os

import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from shared.image_preprocessing import PreprocessingError, preprocess_bytes

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def read_upload(image: UploadFile) -> bytes:
    """Reads the uploaded file, rejecting anything over MAX_UPLOAD_BYTES."""

    too_large = HTTPException(
        status_code=413,
        detail=f"Image is too large (max {MAX_UPLOAD_MB} MB).",
    )

    # Check the size first when it's known, before reading anything
    if image.size is not None and image.size > MAX_UPLOAD_BYTES:
        raise too_large

    # Read one byte past the limit, so an unknown size is still capped
    contents = image.file.read(MAX_UPLOAD_BYTES + 1)

    if len(contents) > MAX_UPLOAD_BYTES:
        raise too_large

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    return contents


def preprocess_image(image: UploadFile) -> torch.Tensor:
    """Uploaded image -> model input tensor [1, 1, 128, 512] (see shared/image_preprocessing.py)."""

    contents = read_upload(image)

    try:
        image_array = preprocess_bytes(contents)
    except PreprocessingError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # Set SAVE_PREPROCESSED_IMAGE=1 to see exactly what the model receives
    if os.environ.get("SAVE_PREPROCESSED_IMAGE"):
        Image.fromarray(image_array).save("debug_preprocessed.png")

    # uint8 [0, 255] -> float32 [0, 1], shaped [batch, channel, height, width]
    image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0)

    return image_tensor.unsqueeze(0).unsqueeze(0)
