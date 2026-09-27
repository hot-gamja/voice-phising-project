"""Model comparison table under the §15 rule, from saved Validation score files.

Run:  uv run python -m scripts.compare_models
Reads experiments/*_val_scores.json, applies: precision at recall >= 0.90,
tie within 2 Validation phishing calls, then F1. Writes experiments/model_comparison_val.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.preprocess import load_processed_split
from src.evaluation.threshold import TARGET_RECALL, metrics_at

SCORE_FILES = ["dlen", "tfidf_logistic", "tfidf_svm", "roberta"]
TIE_CALLS = 2
OUTPUT_PATH = Path("experiments/model_comparison_val.md")


def main() -> None:
    labels = {row["call_id"]: row["label"] for row in load_processed_split("val")}
    positives = sum(labels.values())
    rows = []
    for name in SCORE_FILES:
        path = Path(f"experiments/{name}_val_scores.json")
        if not path.exists():
            continue
        saved = json.load(open(path, encoding="utf-8"))
        call_ids = list(saved["scores"])
        point = metrics_at([labels[c] for c in call_ids], [saved["scores"][c] for c in call_ids], saved["decision_threshold"])
        rows.append((name, point))

    # Tie tolerance: precision difference equivalent to TIE_CALLS phishing calls at this recall.
    predicted_positive = positives * TARGET_RECALL
    tolerance = TIE_CALLS / (predicted_positive + TIE_CALLS)
    best_precision = max(point.precision for _, point in rows)
    tied = [(name, point) for name, point in rows if best_precision - point.precision <= tolerance]
    winner = max(tied, key=lambda item: item[1].f1)[0]

    lines = [f"# Model comparison — Validation (n={len(labels)}, vishing {positives})", "",
             f"Rule (§15): precision at recall ≥ {TARGET_RECALL}; tie if precision within {TIE_CALLS} calls (≈{tolerance:.3f}); then F1.", "",
             "| model | threshold | precision | recall | F1 | note |", "|---|---:|---:|---:|---:|---|"]
    for name, point in rows:
        note = "**selected**" if name == winner else ("tied" if (name, point) in tied else "")
        lines.append(f"| {name} | {point.threshold:+.3f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} | {note} |")
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
