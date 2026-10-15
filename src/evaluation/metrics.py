"""혼동 행렬, 오탐률(FPR), macro F1 — V1이 계산하지 않았던 지표들.

V1은 피싱 클래스 하나만 보는 binary precision/recall/F1을 보고했다. 일반화
연구는 binary F1이 답할 수 없는 두 가지를 더 묻는다: 정상 통화를 피싱이라고
얼마나 자주 오판하는가(오탐률), 그리고 양쪽 클래스를 똑같이 중요하게 볼 때
성능이 얼마인가(macro F1). 둘 다 혼동 행렬 카운트에서 나오므로 그것을 기본
단위로 둔다.

src/evaluation/threshold.py는 experiments/freeze.json으로 봉인돼 있다. 그쪽
binary 지표는 그대로 두고 이 모듈을 옆에 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support

POSITIVE_LABEL = 1  # 피싱


@dataclass
class Confusion:
    """임계값 하나에서의 카운트. 양성=피싱, 음성=정상 통화."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def actual_positives(self) -> int:
        """예측과 무관하게, 집합에 실제로 있는 피싱 통화 수."""
        return self.true_positive + self.false_negative

    @property
    def actual_negatives(self) -> int:
        """예측과 무관하게, 집합에 실제로 있는 정상 통화 수."""
        return self.false_positive + self.true_negative


def predictions_at(scores: list[float], threshold: float) -> list[int]:
    """score >= threshold면 1. threshold.metrics_at와 같은 규칙을 의도적으로 유지한다."""
    return [int(score >= threshold) for score in scores]


def confusion_at(labels: list[int], scores: list[float], threshold: float) -> Confusion:
    """임계값 하나에서의 혼동 행렬 카운트."""
    if len(labels) != len(scores):
        raise ValueError(f"labels {len(labels)}개인데 scores는 {len(scores)}개")
    counts = {(1, 1): 0, (0, 1): 0, (1, 0): 0, (0, 0): 0}
    for label, prediction in zip(labels, predictions_at(scores, threshold)):
        counts[(int(label == POSITIVE_LABEL), prediction)] += 1
    return Confusion(
        true_positive=counts[(1, 1)],
        false_positive=counts[(0, 1)],
        false_negative=counts[(1, 0)],
        true_negative=counts[(0, 0)],
    )


def false_positive_rate(confusion: Confusion) -> float:
    """FP / (FP + TN) — 정상 통화 중 피싱으로 잘못 표시된 비율.

    정상 통화가 0건인 집합에서는 nan을 돌려준다. 틀릴 대상이 애초에 없는데
    0.0을 보고하면 "완벽했다"로 읽히기 때문이다.
    """
    if confusion.actual_negatives == 0:
        return float("nan")
    return confusion.false_positive / confusion.actual_negatives


def macro_f1(labels: list[int], scores: list[float], threshold: float) -> float:
    """두 클래스 F1의 단순 평균(가중치 없음).

    sklearn을 재사용하는 이유: threshold.py:25가 같은 함수를 average="binary"로
    쓰고 있어서, 여기서 average="macro"만 바꾸면 두 지표가 같은 구현에서 나온다.
    직접 유도하면 V1 숫자와 미세하게 어긋날 여지가 생긴다.
    """
    predictions = predictions_at(scores, threshold)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return float(f1)
