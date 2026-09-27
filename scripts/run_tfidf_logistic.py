"""P04 B0: TF-IDF char n-gram + Logistic Regression, Train -> Validation.

Run:  uv run python -m scripts.run_tfidf_logistic
Main comparison uses model_text (Head+Tail). A full-transcript run is
reported once as a reference only (plan §9). Test is not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.data.model_text import load_model_text_split
from src.evaluation.threshold import TARGET_RECALL, metrics_at, observed_score_thresholds, pr_table, select_threshold

C_CANDIDATES = [0.01, 0.1, 1, 10]
MIN_DF = 2  # n-grams seen in only one training call are noise
RESULTS_PATH = Path("experiments/tfidf_logistic_val_results.md")
SCORES_PATH = Path("experiments/tfidf_logistic_val_scores.json")
MODEL_PATH = Path("models/tfidf_logistic.joblib")


def build_pipeline(c_value: float):
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=MIN_DF, sublinear_tf=True),
        LogisticRegression(C=c_value, max_iter=2000, random_state=42),
    )


def fit_and_score(train_rows, val_rows, field: str, c_value: float):
    pipeline = build_pipeline(c_value)
    pipeline.fit([row[field] for row in train_rows], [row["label"] for row in train_rows])
    scores = pipeline.predict_proba([row[field] for row in val_rows])[:, 1].tolist()
    return pipeline, scores


def main() -> None:
    train_rows = load_model_text_split("train")
    val_rows = load_model_text_split("val")
    val_y = [row["label"] for row in val_rows]

    lines = ["# B0 TF-IDF(char_wb 2-3) + Logistic — Train → Validation", "",
             f"- min_df={MIN_DF}, sublinear_tf, C ∈ {C_CANDIDATES}. Selection: precision at recall ≥ {TARGET_RECALL} (§15), tie → F1",
             "", "## C sweep on model_text (Head+Tail)", "",
             "| C | vocab | thr (exact) | precision@R≥0.90 | recall | F1@thr | F1@0.5 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    runs = {}
    for c_value in C_CANDIDATES:
        pipeline, scores = fit_and_score(train_rows, val_rows, "model_text", c_value)
        point = select_threshold(val_y, scores, candidates=observed_score_thresholds(scores))
        vocab_size = len(pipeline.named_steps["tfidfvectorizer"].vocabulary_)
        runs[c_value] = (pipeline, scores, point)
        lines.append(f"| {c_value} | {vocab_size} | {point.threshold:.4f} | {point.precision:.3f} | {point.recall:.3f} "
                     f"| {point.f1:.3f} | {metrics_at(val_y, scores, 0.5).f1:.3f} |")

    best_c = max(runs, key=lambda c: (runs[c][2].precision, runs[c][2].f1))
    best_pipeline, best_scores, best_point = runs[best_c]
    lines += ["", f"**Selected C = {best_c}**, decision_threshold = {best_point.threshold:.4f}, "
              f"precision {best_point.precision:.3f} / recall {best_point.recall:.3f} / F1 {best_point.f1:.3f}", ""]

    lines += ["## PR trade-off (selected C, Validation)", "", "| threshold | precision | recall | F1 |", "|---:|---:|---:|---:|"]
    for point in pr_table(val_y, best_scores):
        lines.append(f"| {point.threshold:.2f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} |")

    _, full_scores = fit_and_score(train_rows, val_rows, "text", best_c)
    full_point = select_threshold(val_y, full_scores, candidates=observed_score_thresholds(full_scores))
    lines += ["", "## Reference only: tfidf_fulltext (full transcript, same C)", "",
              f"precision {full_point.precision:.3f} / recall {full_point.recall:.3f} / F1 {full_point.f1:.3f} at thr {full_point.threshold:.2f}. "
              "Not used for model selection (plan §9)."]

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SCORES_PATH.write_text(json.dumps({"model": "tfidf_logistic", "split": "val", "C": best_c,
                                       "decision_threshold": best_point.threshold,
                                       "scores": {row["call_id"]: s for row, s in zip(val_rows, best_scores)}},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "decision_threshold": best_point.threshold}, MODEL_PATH)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
