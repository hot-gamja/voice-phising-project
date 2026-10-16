"""계획서 §21 WebSocket 검사: 세션, seq 순서, context, 빈 텍스트, 두 세션."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.inference.predictor import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="frozen model artifact not present")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_context_accumulates_and_scores(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "session_id": "s1"})
        ws.send_json({"type": "segment", "session_id": "s1", "seq": 1, "text": "서울중앙지검입니다."})
        first = ws.receive_json()
        ws.send_json({"type": "segment", "session_id": "s1", "seq": 2, "text": "고객님 계좌가 범죄에 연루되었습니다."})
        second = ws.receive_json()
        ws.send_json({"type": "end", "session_id": "s1"})
    assert first["type"] == "score" and first["context_segments"] == 1
    assert second["type"] == "score" and second["context_segments"] == 2
    assert set(second) == {"type", "session_id", "seq", "model_score", "label", "decision_threshold", "context_segments"}


def test_out_of_order_seq_is_error(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "session_id": "s2"})
        ws.send_json({"type": "segment", "session_id": "s2", "seq": 2, "text": "안녕하세요"})
        assert ws.receive_json()["type"] == "score"
        ws.send_json({"type": "segment", "session_id": "s2", "seq": 1, "text": "늦게 온 메시지"})
        error = ws.receive_json()
    assert error["type"] == "error" and "seq" in error["message"]


def test_empty_text_is_error(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "session_id": "s3"})
        ws.send_json({"type": "segment", "session_id": "s3", "seq": 1, "text": "   "})
        error = ws.receive_json()
    assert error["type"] == "error"


def test_segment_without_start_is_error(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "segment", "session_id": "ghost", "seq": 1, "text": "여보세요"})
        assert ws.receive_json()["type"] == "error"


def test_two_sessions_do_not_share_context(client):
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        a.send_json({"type": "start", "session_id": "A"})
        b.send_json({"type": "start", "session_id": "B"})
        a.send_json({"type": "segment", "session_id": "A", "seq": 1, "text": "무엇을 도와드릴까요?"})
        b.send_json({"type": "segment", "session_id": "B", "seq": 1, "text": "서울중앙지검입니다."})
        a.send_json({"type": "segment", "session_id": "A", "seq": 2, "text": "환불 문의드립니다."})
        a1, a2 = a.receive_json(), a.receive_json()
        b1 = b.receive_json()
    assert a2["context_segments"] == 2 and b1["context_segments"] == 1
    assert {a1["session_id"], a2["session_id"]} == {"A"} and b1["session_id"] == "B"
