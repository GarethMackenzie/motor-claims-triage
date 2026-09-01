"""Quality controls for the wholly synthetic source data."""

from __future__ import annotations

import pandas as pd

from src.generate_data import AG_CLAIM_TYPES, MOTOR_CLAIM_TYPES, PROVINCES, generate_claims


def test_generation_is_deterministic() -> None:
    first = generate_claims(n=500, seed=7)
    second = generate_claims(n=500, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_schema_ids_and_nulls() -> None:
    data = generate_claims(n=500, seed=1)
    assert list(data.columns) == [
        "claim_id", "policy_type", "region", "claim_type", "claim_amount",
        "policy_tenure_months", "claimant_age", "prior_claims_count",
        "days_to_report", "police_report_filed", "fraud_flag",
    ]
    assert data["claim_id"].is_unique
    assert data.isna().sum().sum() == 0


def test_numeric_ranges() -> None:
    data = generate_claims(n=1_000, seed=2)
    assert (data["claim_amount"] > 0).all()
    assert data["claimant_age"].between(18, 85).all()
    assert data["policy_tenure_months"].between(1, 240).all()
    assert (data["prior_claims_count"] >= 0).all()
    assert data["days_to_report"].between(0, 60).all()


def test_categories_are_governed() -> None:
    data = generate_claims(n=2_000, seed=3)
    assert set(data["policy_type"]) <= {"Motor", "Agricultural"}
    assert set(data["region"]) <= set(PROVINCES)
    assert set(data["claim_type"]) <= set(MOTOR_CLAIM_TYPES + AG_CLAIM_TYPES)
    assert set(data["fraud_flag"]) <= {0, 1}


def test_positive_label_prevalence_is_within_expected_range() -> None:
    rate = generate_claims(n=8_000, seed=42)["fraud_flag"].mean()
    assert 0.03 <= rate <= 0.07
