"""Leakage-safe feature and preprocessing definitions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "fraud_flag"
ID_COLUMN = "claim_id"
CATEGORICAL_COLUMNS = ["policy_type", "region", "claim_type"]
NUMERIC_COLUMNS = [
    "claim_amount",
    "policy_tenure_months",
    "claimant_age",
    "prior_claims_count",
    "days_to_report",
    "police_report_filed",
]


def feature_columns(include_age: bool = True) -> list[str]:
    """Return governed model inputs in a stable order."""

    numeric = [column for column in NUMERIC_COLUMNS if include_age or column != "claimant_age"]
    return CATEGORICAL_COLUMNS + numeric


def split_features_target(
    data: pd.DataFrame,
    *,
    include_age: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target and identifier."""

    columns = feature_columns(include_age=include_age)
    missing = sorted(set(columns + [TARGET_COLUMN]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data.loc[:, columns].copy(), data[TARGET_COLUMN].astype(int).copy()


def make_preprocessor(*, include_age: bool = True) -> ColumnTransformer:
    """Create transformations that are fitted only inside a model pipeline."""

    numeric = [column for column in NUMERIC_COLUMNS if include_age or column != "claimant_age"]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_model_pipeline(
    estimator: BaseEstimator,
    *,
    include_age: bool = True,
) -> Pipeline:
    """Bind preprocessing and estimation into one cross-validation unit."""

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(include_age=include_age)),
            ("model", estimator),
        ]
    )


def fitted_categories(pipeline: Pipeline) -> dict[str, list[Any]]:
    """Expose fitted training categories for tests and model review."""

    encoder = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )
    return {
        column: list(values)
        for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_, strict=True)
    }
