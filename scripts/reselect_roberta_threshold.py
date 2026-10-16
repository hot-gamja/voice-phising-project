"""저장된 Validation 점수로부터 RoBERTa의 decision_threshold를 다시 계산한다.

실행:  uv run python -m scripts.reselect_roberta_threshold
계획서 §11의 후보 정책이 바뀔 때 재학습 없이 쓴다. lr2e-05 점수 파일, 선택된 LR 복사본,
결과 md, 모델 디렉터리를 갱신한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.preprocess import load_processed_split
from src.evaluation.threshold import TARGET_RECALL, candidate_thresholds, metrics_at, select_threshold

TAG = "lr2e-05"


def main() -> None:
    labels_by_id = {row["call_id"]: row["label"] for row in load_processed_split("val")}
    score_path = Path(f"experiments/roberta_{TAG}_val_scores.json")
    saved = json.loads(score_path.read_text(encoding="utf-8"))
    call_ids = list(saved["scores"])
    scores = [saved["scores"][call_id] for call_id in call_ids]
    labels = [labels_by_id[call_id] for call_id in call_ids]

    point = select_threshold(labels, scores, candidates=candidate_thresholds(scores))
    default = metrics_at(labels, scores, 0.5)
    saved["decision_threshold"] = point.threshold
    score_path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    selected = dict(saved, model=f"roberta_{TAG} (selected LR)")
    Path("experiments/roberta_val_scores.json").write_text(json.dumps(selected, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"models/roberta_{TAG}/decision_threshold.json").write_text(json.dumps({"decision_threshold": point.threshold}), encoding="utf-8")

    results_path = Path(f"experiments/roberta_{TAG}_val_results.md")
    lines = results_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"| Recall ≥ {TARGET_RECALL}"):
            lines[index] = f"| Recall ≥ {TARGET_RECALL} (§11, candidates in [0.05, 0.95]) | {point.threshold:.4f} | {point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} |"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"roberta {TAG}: threshold {point.threshold:.4f} P {point.precision:.3f} R {point.recall:.3f} F1 {point.f1:.3f} (default 0.5: R {default.recall:.3f})")


if __name__ == "__main__":
    main()
