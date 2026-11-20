
"""
SentinelAI - Prediction Request and Response Schemas
"""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """Incoming network-flow features."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="Dictionary containing exactly 70 network-flow features."
    )


class PredictionResponse(BaseModel):
    """Machine learning prediction response."""

    prediction: Literal["BENIGN", "ATTACK"]

    prediction_code: Literal[0, 1]

    attack_probability: float = Field(
        ge=0.0,
        le=1.0
    )

    benign_probability: float = Field(
        ge=0.0,
        le=1.0
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    model: str
