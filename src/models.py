"""Small, purposeful benchmark set for claims triage."""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def candidate_estimators(
    *,
    random_state: int,
    positive_weight: float,
    fast: bool = False,
) -> Mapping[str, BaseEstimator]:
    """Return the governed model comparison set."""

    tree_count = 80 if fast else 300
    return {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_state,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=tree_count,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=tree_count,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        ),
        "Weighted XGBoost": XGBClassifier(
            n_estimators=tree_count,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        ),
    }
