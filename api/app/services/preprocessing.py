from io import BytesIO

import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from shared.config import IMAGE_HEIGHT, IMAGE_WIDTH


def resize_image(image: np.ndarray) -> np.ndarray:
    target_h = IMAGE_HEIGHT
    target_w = IMAGE_WIDTH

    pil_image = Image.fromarray(image.astype(np.uint8))

    original_w, original_h = pil_image.size

    scale = min(
        target_w / original_w,
        target_h / original_h,
    )

    new_w = round(original_w * scale)
    new_h = round(original_h * scale)

    if new_w == 0 or new_h == 0:
        raise HTTPException(
            status_code=400,
            detail="Image aspect ratio is too extreme to preprocess.",
        )

    resized = pil_image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        mode="L",
        size=(target_w, target_h),
        color=255,
    )

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    canvas.paste(resized, (left, top))

    return np.asarray(canvas, dtype=np.uint8)


async def preprocess_image(image: UploadFile) -> torch.Tensor:
    contents = await image.read()

    pil_image = Image.open(BytesIO(contents)).convert("L")

    image_array = np.asarray(
        pil_image,
        dtype=np.uint8,
    )

    image_array = resize_image(image_array)

    # uint8 [0, 255] -> float32 [0, 1]
    image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0)

    # [H, W] -> [1, H, W]
    image_tensor = image_tensor.unsqueeze(0)

    # [1, H, W] -> [1, 1, H, W]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
