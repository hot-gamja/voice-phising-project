"""Serving-side predictor for the frozen V1 model (plan §18, §20).

Applies the same Head+Tail truncation as training, then the saved TF-IDF +
Logistic pipeline. `model_score` is the pipeline's class-1 probability before
any calibration; it is not a real-world probability.
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
# A local tokenizer directory (baked into the Docker image) avoids a network fetch at startup.
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
    """Loads the frozen pipeline once; `predict` is synchronous and thread-safe for reads."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, tokenizer_name: str = DEFAULT_TOKENIZER):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.decision_threshold = float(artifact["decision_threshold"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def model_text(self, text: str) -> str:
        """Same preprocessing path as training: whitespace cleanup + Head+Tail."""
        return truncate_head_tail(clean_text(text), self.tokenizer)

    def score(self, text: str) -> float:
        return float(self.pipeline.predict_proba([self.model_text(text)])[0, 1])

    def predict(self, text: str) -> Prediction:
        if not clean_text(text):
            raise ValueError("empty text")
        model_score = self.score(text)
        label = LABEL_PHISHING if model_score >= self.decision_threshold else LABEL_NORMAL
        return Prediction(label, model_score, self.decision_threshold, MODEL_VERSION)
