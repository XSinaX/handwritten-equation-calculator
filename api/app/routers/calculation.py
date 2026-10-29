import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.app.schemas.calculation_schema import CalculationResponse
from api.app.services.calculator import calculate_expression
from api.app.services.model_loader import load_model
from api.app.services.preprocessing import preprocess_image
from api.app.services.ml import decode_logits

router = APIRouter(
    prefix="/calculate",
    tags=["Calculation"],
)


@router.post("/", response_model=CalculationResponse)
async def calculate_from_image(
    image: UploadFile = File(...),
    imageModel: str = Form(...),
    sequenceModel: str = Form(...),
):
    image_tensor = await preprocess_image(image)

    model = load_model(
        imageModel,
        sequenceModel,
    )

    with torch.inference_mode():
        logits = model(image_tensor)

    equation = decode_logits(logits)
    print("Equation:", repr(equation))

    try:
        answer = calculate_expression(equation)
    except HTTPException as error:
        # Keep the error message the website shows, but also say what was
        # recognised so testing tools can score it (e.g. "5\div0")
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
            headers={"X-Recognized-Equation": equation},
        )

    print("Answer:", answer)

    return CalculationResponse(
        equation=equation,
        answer=answer,
    )
