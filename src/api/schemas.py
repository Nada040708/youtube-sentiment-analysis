from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    comments: List[str] = Field(
        ...,
        min_length=1,
        description="Liste des commentaires à analyser"
    )


class PredictionResult(BaseModel):
    comment: str
    sentiment: str


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    statistics: dict