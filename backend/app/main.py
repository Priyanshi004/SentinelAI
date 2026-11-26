
"""
SentinelAI - Main FastAPI Application

Responsibilities:
1. Initialize the FastAPI application.
2. Load the ML model during startup.
3. Register prediction and traffic API routes.
4. Provide a health-check endpoint.
5. Allow frontend communication through CORS.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.prediction import router as prediction_router
from app.api.traffic import router as traffic_router
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model when the application starts."""

    print("\nStarting SentinelAI backend...")

    try:
        model_service.load_model()
        print("SentinelAI backend is ready.")
        yield

    finally:
        print("\nShutting down SentinelAI backend...")


app = FastAPI(
    title="SentinelAI API",
    description=(
        "AI-powered cybersecurity threat detection "
        "and network traffic classification system."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes
app.include_router(prediction_router)
app.include_router(traffic_router)


@app.get("/")
def root():
    """API welcome endpoint."""

    return {
        "message": "Welcome to SentinelAI API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Check backend and ML model availability."""

    if not model_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded."
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "SentinelAI Random Forest",
        "features_expected": len(
            model_service.get_feature_columns()
        )
    }
