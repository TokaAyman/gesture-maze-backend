# File: gesture-maze-backend/app/schemas.py

from pydantic import BaseModel
from typing import List

class GestureRequest(BaseModel):
    # The frontend should send a flat array of floats (preprocessed feature vector)
    features: List[float]

class GestureResponse(BaseModel):
    label: str
    confidence: float
