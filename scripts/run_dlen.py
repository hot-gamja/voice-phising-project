"""P03 D-LEN: length-only Logistic Regression, Train -> Validation only.

Run:  uv run python -m scripts.run_dlen
Test is NOT touched here (plan §8-B); it is evaluated once at P09.
Writes experiments/dlen_val_results.md and experiments/dlen_val_scores.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.threshold import TARGET_RECALL, metrics_at, pr_table, select_threshold
from src.features.length_features import FEATURE_NAMES, length_features_for_split

RESULTS_PATH = Path("experiments/dlen_val_results.md")
SCORES_PATH = Path("experiments/dlen_val_scores.json")


def main() -> None:
    train_x, train_y, _ = length_features_for_split("train")
    val_x, val_y, val_ids = length_features_for_split("val")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train_x, train_y)
    val_scores = model.predict_proba(val_x)[:, 1].tolist()

    default_point = metrics_at(val_y, val_scores, 0.5)
    chosen_point = select_threshold(val_y, val_scores)
    coefficients = model.named_steps["logisticregression"].coef_[0]

    lines = ["# D-LEN (length-only) — Train → Validation", "",
             f"- features: {', '.join(FEATURE_NAMES)} (full call, before truncation)",
             f"- model: StandardScaler + LogisticRegression, train n={len(train_y)}, val n={len(val_y)} (vishing {sum(val_y)})",
             f"- standardized coefficients: " + ", ".join(f"{name}={coef:+.2f}" for name, coef in zip(FEATURE_NAMES, coefficients)),
             "", "## Operating points", "",
             "| point | threshold | precision | recall | F1 |", "|---|---:|---:|---:|---:|",
             f"| default 0.5 | {default_point.threshold:.2f} | {default_point.precision:.3f} | {default_point.recall:.3f} | {default_point.f1:.3f} |",
             f"| **Recall ≥ {TARGET_RECALL:.2f} (§11)** | {chosen_point.threshold:.2f} | **{chosen_point.precision:.3f}** | {chosen_point.recall:.3f} | {chosen_point.f1:.3f} |",
             "", "## PR trade-off (Validation)", "",
             "| threshold | precision | recall | F1 |", "|---:|---:|---:|---:|"]
    for point in pr_table(val_y, val_scores):
        lines.append(f"| {point.threshold:.2f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} |")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SCORES_PATH.write_text(json.dumps(
        {"model": "dlen", "split": "val", "decision_threshold": chosen_point.threshold,
         "scores": dict(zip(val_ids, val_scores))}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
