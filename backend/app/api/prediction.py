
"""
SentinelAI - Prediction API

Endpoint:
POST /api/predict
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)

from app.services.model_service import model_service


router = APIRouter(
    prefix="/api",
    tags=["Prediction"]
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def predict_network_traffic(
    request: PredictionRequest
):
    """
    Analyze network-flow features and classify traffic
    as BENIGN or ATTACK.
    """

    try:
        result = model_service.predict(
            request.features
        )

        return PredictionResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal server error."
        ) from exc
