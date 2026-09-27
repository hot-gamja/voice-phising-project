"""Validation-based threshold selection (plan §11) and the PR trade-off table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def quantile_thresholds(scores: list[float], steps: int = 19) -> list[float]:
    """Candidate thresholds for scores that are not probabilities (e.g. SVM margins).

    Evenly spaced quantiles of the observed scores, so the grid adapts to the
    score range instead of assuming 0..1.
    """
    quantiles = np.linspace(0.05, 0.95, steps)
    return [round(float(value), 4) for value in np.quantile(scores, quantiles)]


def observed_score_thresholds(scores: list[float]) -> list[float]:
    """Every distinct Validation score as a candidate: no grid loss.

    Any threshold between two observed scores gives the same predictions as the
    higher one, so this set covers all reachable operating points.
    """
    return sorted(set(float(score) for score in scores))


def pr_table(labels: list[int], scores: list[float], candidates: list[float] | None = None) -> list[OperatingPoint]:
    """One row per candidate threshold (default grid 0.05..0.95 for probabilities)."""
    candidates = CANDIDATE_THRESHOLDS if candidates is None else candidates
    return [metrics_at(labels, scores, threshold) for threshold in candidates]


def select_threshold(
    labels: list[int],
    scores: list[float],
    target_recall: float = TARGET_RECALL,
    candidates: list[float] | None = None,
) -> OperatingPoint:
    """Highest-precision threshold whose recall is still >= target_recall.

    Falls back to the lowest candidate threshold if no candidate reaches the target.
    """
    table = pr_table(labels, scores, candidates)
    eligible = [point for point in table if point.recall >= target_recall]
    if not eligible:
        return table[0]
    # Same precision -> prefer more recall (i.e. the lower threshold).
    return max(eligible, key=lambda point: (point.precision, point.recall, -point.threshold))
