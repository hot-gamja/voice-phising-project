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
    """Midpoints between consecutive distinct Validation scores: no grid loss.

    Every reachable operating point is covered, and placing each candidate in
    the middle of the gap (not on an observed score) leaves equal margin on
    both sides, so a small score shift at Test time does not flip a call.
    """
    distinct = sorted(set(float(score) for score in scores))
    return [(low + high) / 2 for low, high in zip(distinct, distinct[1:])]


def restrict_candidates(candidates: list[float], lower: float, upper: float, fallback: float) -> list[float]:
    """Keep candidates inside [lower, upper]; if none remain, use the fallback alone.

    Used to refuse extreme thresholds (e.g. 0.007) that only exist to catch one
    outlier and would be unusable in serving.
    """
    inside = [candidate for candidate in candidates if lower <= candidate <= upper]
    return inside if inside else [fallback]


PROBABILITY_RANGE = (0.05, 0.95, 0.5)  # lower, upper, fallback for 0..1 scores
MARGIN_QUANTILES = (0.05, 0.95, 0.0)   # quantile bounds and fallback for SVM margins


def candidate_thresholds(scores: list[float], score_kind: str = "probability") -> list[float]:
    """V1 policy (plan §11): gap midpoints, restricted away from the extremes.

    score_kind "probability" keeps candidates in [0.05, 0.95] (fallback 0.5);
    "margin" keeps them between the 5th and 95th percentile of the scores
    (fallback 0.0).
    """
    midpoints = observed_score_thresholds(scores)
    if score_kind == "margin":
        lower_q, upper_q, fallback = MARGIN_QUANTILES
        lower, upper = float(np.quantile(scores, lower_q)), float(np.quantile(scores, upper_q))
    else:
        lower, upper, fallback = PROBABILITY_RANGE
    return restrict_candidates(midpoints, lower, upper, fallback)


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
    if candidates is not None and not candidates:
        raise ValueError("select_threshold needs at least one candidate threshold")
    table = pr_table(labels, scores, candidates)
    eligible = [point for point in table if point.recall >= target_recall]
    if not eligible:
        return table[0]
    # Same precision -> prefer more recall (i.e. the lower threshold).
    return max(eligible, key=lambda point: (point.precision, point.recall, -point.threshold))
