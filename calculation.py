import logging
import time

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.app.schemas.calculation_schema import CalculationResponse
from api.app.services.calculator import calculate_expression
from api.app.services.ml import decode_logits
from api.app.services.model_loader import load_model, validate_model_names
from api.app.services.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/calculate",
    tags=["Calculation"],
)


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# CPU-heavy preprocessing and inference don't block other requests.
@router.post("/", response_model=CalculationResponse)
def calculate_from_image(
    image: UploadFile = File(...),
    imageModel: str = Form(...),
    sequenceModel: str = Form(...),
):
    start = time.perf_counter()

    # Check the model names before doing any image work
    validate_model_names(imageModel, sequenceModel)

    image_tensor = preprocess_image(image)

    model = load_model(
        imageModel,
        sequenceModel,
    )

    with torch.inference_mode():
        logits = model(image_tensor)

    equation = decode_logits(logits)

    try:
        answer = calculate_expression(equation)
    except HTTPException as error:
        logger.info("model=%s-%s equation=%r error=%r", imageModel, sequenceModel, equation, error.detail)

        # Keep the error message the website shows, but also say what was
        # recognised so testing tools can score it (e.g. "5\div0")
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
            headers={"X-Recognized-Equation": equation},
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "model=%s-%s equation=%r answer=%s took=%.0fms",
        imageModel,
        sequenceModel,
        equation,
        answer,
        elapsed_ms,
    )

    return CalculationResponse(
        equation=equation,
        answer=answer,
    )
