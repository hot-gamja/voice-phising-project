"""P05 B1: TF-IDF 문자 n-gram + LinearSVC, Train → Validation.

실행: uv run python -m scripts.run_tfidf_svm
LinearSVC에는 predict_proba가 없다; 점수는 decision_function 마진이므로
임계값 그리드는 Validation 점수 분위수로 만든다(계획서 §10). Test는 건드리지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data.model_text import load_model_text_split
from src.evaluation.threshold import TARGET_RECALL, candidate_thresholds, pr_table, quantile_thresholds, select_threshold

C_CANDIDATES = [0.01, 0.1, 1, 10]
MIN_DF = 2
RESULTS_PATH = Path("experiments/tfidf_svm_val_results.md")
SCORES_PATH = Path("experiments/tfidf_svm_val_scores.json")
MODEL_PATH = Path("models/tfidf_svm.joblib")


def build_pipeline(c_value: float):
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=MIN_DF, sublinear_tf=True),
        LinearSVC(C=c_value, random_state=42),
    )


def main() -> None:
    train_rows = load_model_text_split("train")
    val_rows = load_model_text_split("val")
    val_y = [row["label"] for row in val_rows]

    lines = ["# B1 TF-IDF(char_wb 2-3) + LinearSVC — Train → Validation", "",
             f"- scores = decision_function margins (not probabilities). Thresholds = Validation score quantiles. "
             f"Selection: precision at recall ≥ {TARGET_RECALL} (§15), tie → F1", "",
             "## C sweep on model_text (Head+Tail)", "",
             "| C | score range | thr | precision@R≥0.90 | recall | F1@thr | F1@0 |", "|---:|---|---:|---:|---:|---:|---:|"]
    runs = {}
    for c_value in C_CANDIDATES:
        pipeline = build_pipeline(c_value)
        pipeline.fit([row["model_text"] for row in train_rows], [row["label"] for row in train_rows])
        scores = pipeline.decision_function([row["model_text"] for row in val_rows]).tolist()
        candidates = quantile_thresholds(scores)
        point = select_threshold(val_y, scores, candidates=candidate_thresholds(scores, score_kind="margin"))
        zero_point = pr_table(val_y, scores, [0.0])[0]
        runs[c_value] = (pipeline, scores, point, candidates)
        lines.append(f"| {c_value} | {min(scores):+.2f} … {max(scores):+.2f} | {point.threshold:+.3f} | {point.precision:.3f} "
                     f"| {point.recall:.3f} | {point.f1:.3f} | {zero_point.f1:.3f} |")

    best_c = max(runs, key=lambda c: (runs[c][2].precision, runs[c][2].f1))
    best_pipeline, best_scores, best_point, best_candidates = runs[best_c]
    lines += ["", f"**Selected C = {best_c}**, decision_threshold = {best_point.threshold:+.3f} (margin), "
              f"precision {best_point.precision:.3f} / recall {best_point.recall:.3f} / F1 {best_point.f1:.3f}", "",
              "## PR trade-off (selected C, Validation score quantiles)", "",
              "| threshold (margin) | precision | recall | F1 |", "|---:|---:|---:|---:|"]
    for point in pr_table(val_y, best_scores, best_candidates):
        lines.append(f"| {point.threshold:+.3f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} |")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SCORES_PATH.write_text(json.dumps({"model": "tfidf_svm", "split": "val", "C": best_c,
                                       "decision_threshold": best_point.threshold, "score_kind": "margin",
                                       "scores": {row["call_id"]: s for row, s in zip(val_rows, best_scores)}},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best_pipeline, "decision_threshold": best_point.threshold}, MODEL_PATH)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
