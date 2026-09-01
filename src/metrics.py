"""Ranking and capacity metrics for review prioritisation."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def top_k_mask(scores: np.ndarray, capacity: float) -> np.ndarray:
    """Flag exactly the highest-ranked capacity share using stable ordering."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")

    n_flagged = max(1, math.ceil(values.size * capacity))
    ranked = np.argsort(-values, kind="stable")
    mask = np.zeros(values.size, dtype=int)
    mask[ranked[:n_flagged]] = 1
    return mask


def capacity_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    capacity: float,
) -> dict[str, float | int]:
    """Calculate queue performance at one review-capacity operating point."""

    actual = np.asarray(y_true, dtype=int)
    predicted = top_k_mask(np.asarray(scores, dtype=float), capacity)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and scores must have matching shapes")

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    n_flagged = int(predicted.sum())
    positives = int(actual.sum())
    base_rate = float(actual.mean())
    precision = float(tp / n_flagged) if n_flagged else 0.0
    recall = float(tp / positives) if positives else 0.0
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    lift = float(precision / base_rate) if base_rate else 0.0
    return {
        "capacity": float(capacity),
        "n_flagged": n_flagged,
        "precision": precision,
        "recall": recall,
        "false_positives": int(fp),
        "false_positive_rate": false_positive_rate,
        "lift": lift,
        "positives_captured": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def probability_ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate threshold-independent discrimination metrics."""

    actual = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(actual, values)),
        "pr_auc": float(average_precision_score(actual, values)),
    }
