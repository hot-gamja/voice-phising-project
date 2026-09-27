"""Validation-based threshold selection (plan §11) and the PR trade-off table."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support

TARGET_RECALL = 0.90  # V1 operating point. Temporary, not cost-derived (plan §11).
CANDIDATE_THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]  # 0.05 ... 0.95


@dataclass
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    f1: float


def metrics_at(labels: list[int], scores: list[float], threshold: float) -> OperatingPoint:
    """Precision / Recall / F1 for the positive class when score >= threshold."""
    predictions = [int(score >= threshold) for score in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return OperatingPoint(threshold, float(precision), float(recall), float(f1))


def pr_table(labels: list[int], scores: list[float]) -> list[OperatingPoint]:
    """One row per candidate threshold."""
    return [metrics_at(labels, scores, threshold) for threshold in CANDIDATE_THRESHOLDS]


def select_threshold(labels: list[int], scores: list[float], target_recall: float = TARGET_RECALL) -> OperatingPoint:
    """Highest-precision threshold whose recall is still >= target_recall.

    Falls back to the lowest candidate threshold if no candidate reaches the target.
    """
    eligible = [point for point in pr_table(labels, scores) if point.recall >= target_recall]
    if not eligible:
        return metrics_at(labels, scores, CANDIDATE_THRESHOLDS[0])
    return max(eligible, key=lambda point: (point.precision, point.threshold))
