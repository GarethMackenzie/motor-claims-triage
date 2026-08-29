"""
Sanity checks for the synthetic data generator. Run with:
    pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import generate_claims  # noqa: E402


def test_shape_and_columns():
    df = generate_claims(n=500, seed=1)
    assert len(df) == 500
    expected_cols = {
        "claim_id", "policy_type", "region", "claim_type", "claim_amount",
        "policy_tenure_months", "claimant_age", "prior_claims_count",
        "days_to_report", "police_report_filed", "fraud_flag",
    }
    assert expected_cols.issubset(set(df.columns))


def test_no_nulls():
    df = generate_claims(n=500, seed=1)
    assert df.isnull().sum().sum() == 0


def test_fraud_rate_in_plausible_range():
    df = generate_claims(n=5000, seed=1)
    rate = df["fraud_flag"].mean()
    assert 0.01 < rate < 0.12, f"fraud rate {rate:.2%} outside plausible range"


def test_claim_amounts_positive():
    df = generate_claims(n=500, seed=1)
    assert (df["claim_amount"] > 0).all()


def test_deterministic_with_seed():
    df1 = generate_claims(n=200, seed=7)
    df2 = generate_claims(n=200, seed=7)
    assert df1.equals(df2)


def test_policy_types_valid():
    df = generate_claims(n=500, seed=1)
    assert set(df["policy_type"].unique()).issubset({"Motor", "Agricultural"})
