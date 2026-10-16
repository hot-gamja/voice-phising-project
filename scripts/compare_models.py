"""§15 규칙에 따른 모델 비교 표, 저장된 Validation 점수 파일로부터 생성.

실행:  uv run python -m scripts.compare_models
experiments/*_val_scores.json을 읽어서, recall >= 0.90에서의 precision을 적용하고,
2개 Validation 피싱 통화 이내면 동률 처리 후 F1으로 비교한다. experiments/model_comparison_val.md를 쓴다.
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

    # 동률 허용 범위: 이 recall에서 TIE_CALLS 피싱 통화에 해당하는 precision 차이.
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
