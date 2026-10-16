"""freeze된 V1 모델의 서빙 측 predictor (계획서 §18, §20).

학습과 똑같은 Head+Tail 절단을 적용한 뒤 저장된 TF-IDF + Logistic 파이프라인에
넣는다. `model_score`는 보정을 거치지 않은 파이프라인의 클래스 1 확률이며,
실제 피싱 확률이 아니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
from transformers import AutoTokenizer

from src.data.model_text import TOKENIZER_NAME
from src.data.preprocess import clean_text
from src.data.truncate import truncate_head_tail

MODEL_VERSION = "v1"
DEFAULT_MODEL_PATH = Path("models/tfidf_logistic.joblib")
# tokenizer를 로컬 디렉터리(Docker 이미지에 구워 넣음)에서 읽어 기동 시 네트워크 요청을 없앤다.
DEFAULT_TOKENIZER = os.environ.get("TOKENIZER_PATH", TOKENIZER_NAME)
LABEL_PHISHING = "PHISHING"
LABEL_NORMAL = "NORMAL"


@dataclass(frozen=True)
class Prediction:
    label: str
    model_score: float
    decision_threshold: float
    model_version: str


class Predictor:
    """freeze된 파이프라인을 한 번만 로드한다. `predict`는 동기 호출이고 읽기에는 스레드 안전하다."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, tokenizer_name: str = DEFAULT_TOKENIZER):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.decision_threshold = float(artifact["decision_threshold"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def model_text(self, text: str) -> str:
        """학습과 같은 전처리 경로: 공백 정리 + Head+Tail 절단."""
        return truncate_head_tail(clean_text(text), self.tokenizer)

    def score(self, text: str) -> float:
        return float(self.pipeline.predict_proba([self.model_text(text)])[0, 1])

    def predict(self, text: str) -> Prediction:
        if not clean_text(text):
            raise ValueError("empty text")
        model_score = self.score(text)
        label = LABEL_PHISHING if model_score >= self.decision_threshold else LABEL_NORMAL
        return Prediction(label, model_score, self.decision_threshold, MODEL_VERSION)
