from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    probability: float
    inference_time_ms: float
