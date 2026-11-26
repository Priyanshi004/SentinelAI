"""
SentinelAI - Network Traffic API

Endpoints for simulated network traffic monitoring.
"""

from fastapi import APIRouter, HTTPException

from app.services.traffic_service import traffic_service


router = APIRouter(
    prefix="/api/traffic",
    tags=["Network Traffic"]
)


@router.post("/start")
def start_traffic_simulation():
    """Start simulated network traffic."""

    try:
        return {
            "message": "Traffic simulation started.",
            "data": traffic_service.start()
        }

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc


@router.post("/stop")
def stop_traffic_simulation():
    """Stop simulated network traffic."""

    return {
        "message": "Traffic simulation stopped.",
        "data": traffic_service.stop()
    }


@router.get("/status")
def get_traffic_status():
    """Get current traffic statistics."""

    return traffic_service.get_status()


@router.get("/next")
def process_next_traffic_flow():
    """Process the next simulated network flow."""

    try:
        event = traffic_service.process_next_flow()

        if event is None:
            return {
                "message": "Traffic simulation is stopped.",
                "data": None
            }

        return {
            "message": "Traffic flow processed successfully.",
            "data": event
        }

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc
