"""Cross-validation and holdout evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .metrics import capacity_metrics, probability_ranking_metrics


def cross_validate_pipeline(
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    folds: int,
    random_state: int,
    capacity: float = 0.10,
) -> tuple[dict[str, float | str], list[dict[str, float | int | str]]]:
    """Evaluate a complete preprocessing-model pipeline on development data."""

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    records: list[dict[str, float | int | str]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, target), start=1
    ):
        fitted = clone(pipeline)
        train_features = features.iloc[train_index]
        train_target = target.iloc[train_index]
        validation_features = features.iloc[validation_index]
        validation_target = target.iloc[validation_index]
        fitted.fit(train_features, train_target)
        scores = fitted.predict_proba(validation_features)[:, 1]
        ranking = probability_ranking_metrics(validation_target.to_numpy(), scores)
        operating = capacity_metrics(validation_target.to_numpy(), scores, capacity)
        records.append(
            {
                "Model": model_name,
                "Fold": fold,
                "ROC-AUC": ranking["roc_auc"],
                "PR-AUC": ranking["pr_auc"],
                "Precision@10%": operating["precision"],
                "Recall@10%": operating["recall"],
                "Lift@10%": operating["lift"],
            }
        )

    frame = pd.DataFrame(records)
    summary: dict[str, float | str] = {"Model": model_name}
    for metric in ["ROC-AUC", "PR-AUC", "Precision@10%", "Recall@10%", "Lift@10%"]:
        summary[metric] = float(frame[metric].mean())
        summary[f"{metric} Std"] = float(frame[metric].std(ddof=1))
    return summary, records


def evaluate_holdout(
    target: pd.Series | np.ndarray,
    scores: np.ndarray,
    capacities: Iterable[float],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Evaluate a fitted final model once on the untouched holdout set."""

    actual = np.asarray(target, dtype=int)
    ranking = probability_ranking_metrics(actual, scores)
    summary: dict[str, float | int] = {
        "n_holdout": int(actual.size),
        "positive_labels": int(actual.sum()),
        "positive_rate": float(actual.mean()),
        **ranking,
    }
    operating_points = [capacity_metrics(actual, scores, capacity) for capacity in capacities]
    return summary, operating_points
