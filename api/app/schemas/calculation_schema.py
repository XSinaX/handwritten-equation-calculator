from pydantic import BaseModel


class CalculationResponse(BaseModel):
    equation: str
    answer: float
