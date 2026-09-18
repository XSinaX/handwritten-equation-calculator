from pydantic import BaseModel


class CalculationResponse(BaseModel):
    expression: str
    result: float
