"""계획서 §24 freeze된 TF-IDF Logistic artifact에 대한 모델 추론 테스트."""

import pytest

from src.inference.predictor import DEFAULT_MODEL_PATH, LABEL_NORMAL, LABEL_PHISHING, Predictor

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="frozen model artifact not present")


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_threshold_matches_freeze(predictor):
    assert abs(predictor.decision_threshold - 0.4125) < 1e-3


def test_prediction_fields(predictor):
    prediction = predictor.predict("서울중앙지검입니다. 고객님 계좌가 범죄에 연루되어 안전계좌로 이체하셔야 합니다.")
    assert prediction.label in (LABEL_PHISHING, LABEL_NORMAL)
    assert 0.0 <= prediction.model_score <= 1.0
    assert prediction.model_version == "v1"
    assert prediction.label == (LABEL_PHISHING if prediction.model_score >= prediction.decision_threshold else LABEL_NORMAL)


def test_empty_text_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict("   ")


def test_long_text_is_truncated_like_training(predictor):
    long_text = "고객님 계좌로 삼백만 원을 지금 송금해 주셔야 합니다. " * 300
    model_text = predictor.model_text(long_text)
    assert len(predictor.tokenizer(model_text)["input_ids"]) <= 512
    assert model_text.startswith("고객님 계좌로")
