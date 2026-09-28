"""REST and WebSocket message shapes (plan §18, §19)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_loaded: bool


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Call transcript (any length; Head+Tail applied server-side)")


class PredictResponse(BaseModel):
    label: Literal["PHISHING", "NORMAL"]
    model_score: float = Field(description="Raw model output before calibration. Not a real-world probability.")
    decision_threshold: float
    model_version: str


class WsClientMessage(BaseModel):
    type: Literal["start", "segment", "end"]
    session_id: str = Field(min_length=1)
    seq: int | None = None
    text: str | None = None


class WsScoreMessage(BaseModel):
    type: Literal["score"] = "score"
    session_id: str
    seq: int
    model_score: float
    label: Literal["PHISHING", "NORMAL"]
    decision_threshold: float
    context_segments: int


class WsErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    session_id: str
    seq: int | None = None
    message: str
