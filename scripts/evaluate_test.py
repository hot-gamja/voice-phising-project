"""P09 2단계: Test Unlock. freeze된 모든 모델을 공식 Test에 한 번 평가한다.

실행:  uv run python -m scripts.evaluate_test
experiments/freeze.json이 필요하다. 각 모델의 Validation에서 선택된 임계값을 사용한다.
experiments/p09_test_results.md와 experiments/test_scores.json을 쓴다.
리포트 전용: 여기서는 아무것도 튜닝하지 않는다(계획서 §12).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.load_korccvi import load_call_manifest, load_transcript
from src.data.model_text import load_model_text_split
from src.evaluation.threshold import metrics_at
from src.features.length_features import length_features

FREEZE_PATH = Path("experiments/freeze.json")
RESULTS_PATH = Path("experiments/p09_test_results.md")
SCORES_PATH = Path("experiments/test_scores.json")
BOOTSTRAP_ROUNDS = 1000


def roberta_scores(model_dir: Path, texts: list[str]) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(texts[start:start + 16], truncation=True, max_length=512, padding=True, return_tensors="pt")
            logits = model(**batch).logits
            scores += torch.softmax(logits, dim=-1)[:, 1].tolist()
    return scores


def bootstrap_ci(labels: list[int], scores: list[float], threshold: float, metric: str) -> tuple[float, float]:
    """하나의 지표에 대한 95% 퍼센타일 부트스트랩 구간(계획서 §8-A, 선택 사항)."""
    generator = random.Random(42)
    values = []
    n = len(labels)
    for _ in range(BOOTSTRAP_ROUNDS):
        indices = [generator.randrange(n) for _ in range(n)]
        point = metrics_at([labels[i] for i in indices], [scores[i] for i in indices], threshold)
        values.append(getattr(point, metric))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    if not FREEZE_PATH.exists():
        raise SystemExit("No freeze.json. Run scripts.freeze first (plan §8-A: Freeze before Test Unlock).")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if SCORES_PATH.exists():
        raise SystemExit(f"{SCORES_PATH} exists: Test was already evaluated once. Do not re-run.")

    test_rows = load_model_text_split("test")
    labels = [row["label"] for row in test_rows]
    call_ids = [row["call_id"] for row in test_rows]
    model_texts = [row["model_text"] for row in test_rows]
    transcripts = {record["call_id"]: load_transcript(record["transcript_filepath"]) for record in load_call_manifest("test")}
    dlen_x = [length_features(transcripts[call_id]) for call_id in call_ids]

    all_scores: dict[str, list[float]] = {}
    for name, info in freeze["models"].items():
        path = Path(info["path"])
        if name == "dlen":
            all_scores[name] = joblib.load(path)["pipeline"].predict_proba(dlen_x)[:, 1].tolist()
        elif name == "tfidf_logistic":
            all_scores[name] = joblib.load(path)["pipeline"].predict_proba(model_texts)[:, 1].tolist()
        elif name == "tfidf_svm":
            all_scores[name] = joblib.load(path)["pipeline"].decision_function(model_texts).tolist()
        elif name == "roberta":
            all_scores[name] = roberta_scores(path, model_texts)

    positives = sum(labels)
    lines = [f"# P09 Test results — evaluated once (frozen at {freeze['frozen_at']}, commit {freeze['git_commit']})", "",
             f"Test n={len(labels)}, vishing {positives}, normal {len(labels) - positives}. Thresholds are the Validation-chosen values. Report only.", "",
             "| model | threshold | precision | recall | F1 | FP | FN | F1 95% CI (bootstrap) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    errors: dict[str, dict[str, list]] = {}
    for name, scores in all_scores.items():
        threshold = freeze["models"][name]["decision_threshold"]
        point = metrics_at(labels, scores, threshold)
        predictions = [int(score >= threshold) for score in scores]
        fp = [call_ids[i] for i in range(len(labels)) if predictions[i] == 1 and labels[i] == 0]
        fn = [call_ids[i] for i in range(len(labels)) if predictions[i] == 0 and labels[i] == 1]
        errors[name] = {"FP": fp, "FN": fn}
        low, high = bootstrap_ci(labels, scores, threshold, "f1")
        marker = " **(V1 final)**" if name == freeze["final_model"] else ""
        lines.append(f"| {name}{marker} | {threshold:+.3f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} | {len(fp)} | {len(fn)} | {low:.3f} – {high:.3f} |")

    lines += ["", "## Errors per model (call_id, score, chars)", ""]
    chars = {row["call_id"]: len(row["text"]) for row in test_rows}
    score_by = {name: dict(zip(call_ids, scores)) for name, scores in all_scores.items()}
    for name, kinds in errors.items():
        for kind, ids in kinds.items():
            if ids:
                lines.append(f"- {name} {kind} ({len(ids)}): " + ", ".join(f"{c} ({score_by[name][c]:+.3f}, {chars[c]}자)" for c in ids))
    tfidf_wrong = set(errors["tfidf_logistic"]["FP"] + errors["tfidf_logistic"]["FN"])
    roberta_wrong = set(errors["roberta"]["FP"] + errors["roberta"]["FN"])
    lines += ["", "## TF-IDF vs RoBERTa disagreement (plan §12)", "",
              f"- wrong only in TF-IDF: {sorted(tfidf_wrong - roberta_wrong)}",
              f"- wrong only in RoBERTa: {sorted(roberta_wrong - tfidf_wrong)}",
              f"- wrong in both: {sorted(tfidf_wrong & roberta_wrong)}"]

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SCORES_PATH.write_text(json.dumps({"split": "test", "frozen_at": freeze["frozen_at"], "labels": dict(zip(call_ids, labels)),
                                       "scores": {name: dict(zip(call_ids, scores)) for name, scores in all_scores.items()}},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
