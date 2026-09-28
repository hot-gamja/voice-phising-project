"""Async wrapper around the synchronous Predictor (plan §18 구현 관행)."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from src.inference.predictor import Prediction, Predictor


async def predict_async(predictor: Predictor, text: str) -> Prediction:
    """Run the blocking model call off the event loop so other sessions keep flowing."""
    return await run_in_threadpool(predictor.predict, text)
