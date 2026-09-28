"""FastAPI app: REST inference for the frozen V1 model (plan §18).

Run:  uv run uvicorn app.main:app --reload
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
    """Load the model once at startup (plan §18: lifespan, not per request)."""
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.predictor = Predictor(model_path)
    app.state.sessions = SessionStore()
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


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket) -> None:
    """Mock realtime stream (plan §19-21).

    Client -> {"type": "start"|"segment"|"end", "session_id", "seq"?, "text"?}
    Server -> {"type": "score", session_id, seq, model_score, label, decision_threshold, context_segments}
              {"type": "error", session_id, seq?, message}
    Context accumulates per session; each segment scores the whole context so far.
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
            except Exception as error:  # model failure must not kill the socket (plan §21)
                await websocket.send_json(WsErrorMessage(session_id=message.session_id, seq=message.seq, message=f"inference failed: {error}").model_dump())
                continue
            await websocket.send_json(WsScoreMessage(
                session_id=message.session_id, seq=message.seq, model_score=prediction.model_score,
                label=prediction.label, decision_threshold=prediction.decision_threshold,
                context_segments=len(session.segments),
            ).model_dump())
    except WebSocketDisconnect:
        return
