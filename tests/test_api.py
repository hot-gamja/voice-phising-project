"""계획서 §24 REST API 테스트, TestClient 사용 (lifespan은 `with` 안에서 실행)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.inference.predictor import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="frozen model artifact not present")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_version": "v1", "model_loaded": True}


def test_predict_returns_contract_fields(client):
    response = client.post("/predict", json={"text": "서울중앙지검입니다. 고객님 명의 계좌가 범죄에 연루되어 안전계좌로 이체하셔야 합니다."})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "model_score", "decision_threshold", "model_version"}
    assert body["label"] == ("PHISHING" if body["model_score"] >= body["decision_threshold"] else "NORMAL")


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422
