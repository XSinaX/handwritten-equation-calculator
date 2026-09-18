from fastapi import APIRouter, UploadFile, File

from schemas.calculation_shema import CalculationResponse
from services.preprocessing import preprocess_image
from services.prediction import predict_expression
from services.calculator import calculate_expression

router = APIRouter(
    prefix="/calculate",
    tags=["Calculation"],
)


@router.post("/", response_model=CalculationResponse)
async def calculate_from_image(image: UploadFile = File(...)):
    image_data = await image.read()

    vector = preprocess_image(image_data)

    expression = predict_expression(vector)

    result = calculate_expression(expression)

    return CalculationResponse(
        expression=expression,
        result=result,
    )
