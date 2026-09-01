"""Professional evaluation charts generated from holdout scores."""

from __future__ import annotations

import os
from pathlib import Path

MATPLOTLIB_CACHE = Path(__file__).resolve().parent.parent / ".matplotlib"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

BLUE = "#0B5CAD"
TEAL = "#117A8B"
ORANGE = "#D97706"
GREY = "#6B7280"


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_roc(target: np.ndarray, scores: np.ndarray, path: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(target, scores)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(false_positive_rate, true_positive_rate, color=BLUE, linewidth=2, label="Final model")
    plt.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="Random baseline")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Holdout ROC curve")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _finish(path)


def plot_precision_recall(target: np.ndarray, scores: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(target, scores)
    base_rate = float(np.mean(target))
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(recall, precision, color=TEAL, linewidth=2, label="Final model")
    plt.axhline(base_rate, color=GREY, linestyle="--", label="Positive-class base rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Holdout precision-recall curve")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _finish(path)


def _gains(target: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_target = target[order]
    population = np.arange(1, len(target) + 1) / len(target)
    positives = max(1, int(target.sum()))
    gains = np.cumsum(sorted_target) / positives
    return population, gains


def plot_cumulative_gains(target: np.ndarray, scores: np.ndarray, path: Path) -> None:
    population, gains = _gains(target, scores)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(population, gains, color=BLUE, linewidth=2, label="Final model")
    plt.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="Random baseline")
    plt.xlabel("Share of claims reviewed")
    plt.ylabel("Share of positive labels captured")
    plt.title("Holdout cumulative gains")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _finish(path)


def plot_lift(target: np.ndarray, scores: np.ndarray, path: Path) -> None:
    population, gains = _gains(target, scores)
    lift = gains / population
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(population, lift, color=ORANGE, linewidth=2, label="Final model")
    plt.axhline(1.0, color=GREY, linestyle="--", label="Random baseline")
    plt.xlabel("Share of claims reviewed")
    plt.ylabel("Lift over random review")
    plt.title("Holdout lift curve")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    _finish(path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    ordered = comparison.sort_values("PR-AUC")
    plt.figure(figsize=(8.2, 4.8))
    plt.barh(
        ordered["Model"],
        ordered["PR-AUC"],
        xerr=ordered["PR-AUC Std"],
        color=BLUE,
        alpha=0.9,
        capsize=3,
    )
    plt.xlabel("Cross-validated PR-AUC (mean with standard deviation)")
    plt.title("Model comparison on development folds")
    plt.grid(axis="x", alpha=0.2)
    _finish(path)


def plot_permutation_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("importance_mean").tail(10)
    plt.figure(figsize=(8.2, 4.8))
    plt.barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color=TEAL,
        alpha=0.9,
        capsize=3,
    )
    plt.xlabel("Decrease in holdout PR-AUC after permutation")
    plt.title("Permutation importance")
    plt.grid(axis="x", alpha=0.2)
    _finish(path)
