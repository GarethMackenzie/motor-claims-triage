"""Shared experiment settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible settings for model development and final evaluation."""

    random_state: int = 42
    n_claims: int = 8_000
    holdout_fraction: float = 0.20
    cv_folds: int = 5
    capacities: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    permutation_repeats: int = 10


DEFAULT_CONFIG = ExperimentConfig()
