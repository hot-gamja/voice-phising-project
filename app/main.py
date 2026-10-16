"""FastAPI 앱: freeze된 V1 모델의 REST 추론 (계획서 §18).

실행:  uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.schemas import HealthResponse, PredictRequest, PredictResponse, WsClientMessage, WsErrorMessage, WsScoreMessage
from app.service import predict_async
from app.session import SessionStore
from src.inference.predictor import DEFAULT_MODEL_PATH, MODEL_VERSION, Predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """모델을 기동 시 한 번만 로드한다 (계획서 §18: 요청마다가 아니라 lifespan에서)."""
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.predictor = Predictor(model_path)
    app.state.sessions = SessionStore()
    yield
    app.state.predictor = None


app = FastAPI(title="Voice Phishing Detection V1", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """응답: {"status": "ok", "model_version": "v1", "model_loaded": true}."""
    return HealthResponse(status="ok", model_version=MODEL_VERSION, model_loaded=app.state.predictor is not None)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """요청: {"text": "..."}.  응답: label, model_score, decision_threshold, model_version.

    빈 문자열이나 공백뿐인 text는 422 (pydantic + ValueError), 모델이 로드되지 않았으면 503.
    """
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        prediction = await predict_async(app.state.predictor, request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictResponse(**prediction.__dict__)


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket) -> None:
    """실시간 스트림 모의 구현 (계획서 §19-21).

    클라이언트 -> {"type": "start"|"segment"|"end", "session_id", "seq"?, "text"?}
    서버       -> {"type": "score", session_id, seq, model_score, label, decision_threshold, context_segments}
                  {"type": "error", session_id, seq?, message}
    맥락은 세션별로 누적되고, 구간이 들어올 때마다 그때까지의 맥락 전체를 채점한다.
    """
    await websocket.accept()
    sessions: SessionStore = app.state.sessions
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                message = WsClientMessage.model_validate(raw)
            except ValidationError as error:
                await websocket.send_json(WsErrorMessage(session_id=str(raw.get("session_id", "?")), seq=raw.get("seq"),
                                                         message=f"invalid message: {error.errors()[0]['msg']}").model_dump())
                continue

            if message.type == "start":
                sessions.start(message.session_id)
                continue
            if message.type == "end":
                sessions.end(message.session_id)
                continue

            session = sessions.get(message.session_id)
            if session is None:
                await websocket.send_json(WsErrorMessage(session_id=message.session_id, seq=message.seq, message="unknown session; send start first").model_dump())
                continue
            if message.seq is None or not message.text or not message.text.strip():
                await websocket.send_json(WsErrorMessage(session_id=message.session_id, seq=message.seq, message="empty text or missing seq").model_dump())
                continue
            try:
                session.append(message.seq, message.text.strip())
            except ValueError as error:
                await websocket.send_json(WsErrorMessage(session_id=message.session_id, seq=message.seq, message=str(error)).model_dump())
                continue
            try:
                prediction = await predict_async(app.state.predictor, session.context_text())
            except Exception as error:  # 모델이 실패해도 소켓은 살려둬야 한다 (계획서 §21)
                await websocket.send_json(WsErrorMessage(session_id=message.session_id, seq=message.seq, message=f"inference failed: {error}").model_dump())
                continue
            await websocket.send_json(WsScoreMessage(
                session_id=message.session_id, seq=message.seq, model_score=prediction.model_score,
                label=prediction.label, decision_threshold=prediction.decision_threshold,
                context_segments=len(session.segments),
            ).model_dump())
    except WebSocketDisconnect:
        return
