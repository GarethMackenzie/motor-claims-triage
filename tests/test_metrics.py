"""Tests for ranking and capacity calculations."""

from __future__ import annotations

import numpy as np

from src.metrics import capacity_metrics, top_k_mask


def test_top_k_selector_flags_expected_count() -> None:
    scores = np.arange(20, dtype=float)
    mask = top_k_mask(scores, 0.10)
    assert int(mask.sum()) == 2
    assert set(np.flatnonzero(mask)) == {18, 19}


def test_capacity_metrics_reconcile_and_are_correct() -> None:
    target = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = capacity_metrics(target, scores, 0.20)
    assert result["n_flagged"] == 2
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["lift"] == 2.5
    assert result["positives_captured"] == 1
    assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == len(target)
    assert result["fp"] + result["tp"] == result["n_flagged"]


def test_stable_tie_breaking() -> None:
    mask = top_k_mask(np.ones(10), 0.20)
    assert set(np.flatnonzero(mask)) == {0, 1}
