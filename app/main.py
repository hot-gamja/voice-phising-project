"""FastAPI app: REST inference for the frozen V1 model (plan §18).

Run:  uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.service import predict_async
from src.inference.predictor import DEFAULT_MODEL_PATH, MODEL_VERSION, Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (plan §18: lifespan, not per request)."""
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.predictor = Predictor(model_path)
    yield
    app.state.predictor = None


app = FastAPI(title="Voice Phishing Detection V1", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Response: {"status": "ok", "model_version": "v1", "model_loaded": true}."""
    return HealthResponse(status="ok", model_version=MODEL_VERSION, model_loaded=app.state.predictor is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Request: {"text": "..."}.  Response: label, model_score, decision_threshold, model_version.

    422 for an empty/whitespace-only text (pydantic + ValueError), 503 if the model is not loaded.
    """
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        prediction = await predict_async(app.state.predictor, request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictResponse(**prediction.__dict__)
