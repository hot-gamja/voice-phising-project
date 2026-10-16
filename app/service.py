"""동기 Predictor를 async로 감싸는 래퍼 (계획서 §18 구현 관행)."""

from __future__ import annotations

from fastapi.concurrency import run_in_threadpool

from src.inference.predictor import Prediction, Predictor


async def predict_async(predictor: Predictor, text: str) -> Prediction:
    """블로킹 모델 호출을 이벤트 루프 밖으로 빼서 다른 세션이 막히지 않게 한다."""
    return await run_in_threadpool(predictor.predict, text)
