from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.app.services.calculator import calculate_expression
from api.app.services.ctc_inference import CTC_MODELS, recognize_ctc
from api.app.services.segmentation_inference import (
    SEGMENTATION_MODELS,
    preprocess_segmentation_image,
    recognize_segmentation,
)
from ml.preprocessing import preprocess_image

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/calculate")
async def calculate(image: UploadFile = File(...), model: str = Form(...), architecture: str | None = Form(None)):
    """Recognize an uploaded equation with the selected model and calculate it."""
    if model not in CTC_MODELS and model not in SEGMENTATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    selected_architecture = "end_to_end" if model in CTC_MODELS else "segmentation"
    if architecture is not None and architecture != selected_architecture:
        raise HTTPException(status_code=400, detail="Recognition architecture does not match the selected model")

    try:
        with Image.open(BytesIO(await image.read())) as uploaded:
            has_alpha = "A" in uploaded.getbands() or "transparency" in uploaded.info
            source = np.asarray(uploaded.convert("RGBA" if has_alpha else "RGB"), dtype=np.uint8)
        processed = (
            preprocess_image(source) if selected_architecture == "end_to_end" else preprocess_segmentation_image(source)
        )
    except (UnidentifiedImageError, OSError, ValueError, cv2.error) as error:
        raise HTTPException(status_code=400, detail=f"Invalid equation image: {error}") from error

    if model in CTC_MODELS:
        equation = recognize_ctc(processed, model)
    else:
        equation = recognize_segmentation(processed, model)

    try:
        answer = calculate_expression(equation)
    except HTTPException as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
            headers={"X-Recognized-Equation": equation},
        ) from error

    return {"model": model, "equation": equation, "answer": answer}
