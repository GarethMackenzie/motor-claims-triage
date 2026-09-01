"""Leakage, robustness and reproducibility checks for the model pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.features import TARGET_COLUMN, fitted_categories, make_model_pipeline, split_features_target
from src.generate_data import generate_claims


def test_target_and_identifier_are_excluded() -> None:
    features, target = split_features_target(generate_claims(n=200, seed=8))
    assert TARGET_COLUMN not in features.columns
    assert "claim_id" not in features.columns
    assert len(features) == len(target)


def test_preprocessor_learns_categories_from_training_only() -> None:
    data = generate_claims(n=300, seed=9)
    training = data.iloc[:250].copy()
    validation = data.iloc[250:].copy()
    validation["region"] = "Unseen Region"
    training_x, training_y = split_features_target(training)
    validation_x, _ = split_features_target(validation)
    pipeline = make_model_pipeline(DummyClassifier(strategy="prior"))
    pipeline.fit(training_x, training_y)
    assert "Unseen Region" not in fitted_categories(pipeline)["region"]
    scores = pipeline.predict_proba(validation_x)[:, 1]
    assert len(scores) == len(validation)
    assert np.isfinite(scores).all()


def test_model_trains_and_outputs_finite_scores() -> None:
    features, target = split_features_target(generate_claims(n=500, seed=10))
    pipeline = make_model_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42)
    )
    pipeline.fit(features, target)
    scores = pipeline.predict_proba(features.head(25))[:, 1]
    assert scores.shape == (25,)
    assert np.isfinite(scores).all()


def test_fixed_random_state_produces_stable_core_scores() -> None:
    features, target = split_features_target(generate_claims(n=600, seed=11))
    estimator = LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42)
    first = make_model_pipeline(estimator)
    second = make_model_pipeline(estimator)
    first.fit(features, target)
    second.fit(features, target)
    np.testing.assert_allclose(
        first.predict_proba(features)[:, 1],
        second.predict_proba(features)[:, 1],
        rtol=0,
        atol=1e-12,
    )
