import os

import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from shared.image_preprocessing import PreprocessingError, preprocess_bytes


async def preprocess_image(image: UploadFile) -> torch.Tensor:
    """Uploaded image -> model input tensor [1, 1, 128, 512] (see shared/image_preprocessing.py)."""

    contents = await image.read()

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
