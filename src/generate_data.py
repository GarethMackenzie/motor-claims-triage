"""
Synthetic motor & agricultural insurance claims generator.

Portfolio/demo dataset only. Every record here is synthetically generated
from the distributions below - no real policyholder, claimant, or employer
data is used anywhere in this repository.

Fraud propensity is built from a small set of documented risk signals
(reporting delay, no police report, short policy tenure, high claim
value, prior claims count) combined with random noise, so the resulting
classification problem is realistic and NOT perfectly separable - a
model that gets everything right on this dataset would indicate a bug,
not a good model.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "synthetic_claims.csv"

RANDOM_SEED = DEFAULT_CONFIG.random_state
N_CLAIMS = DEFAULT_CONFIG.n_claims
BASE_FRAUD_RATE = 0.045  # ~4.5%, a plausible order-of-magnitude for short-term insurance fraud rates

PROVINCES = [
    "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
    "Free State", "Mpumalanga", "Limpopo", "North West", "Northern Cape",
]
PROVINCE_WEIGHTS = [0.26, 0.13, 0.20, 0.11, 0.07, 0.08, 0.06, 0.06, 0.03]
MOTOR_CLAIM_TYPES = ["Collision", "Theft", "Fire", "Windscreen", "Hijacking", "Third Party"]
AG_CLAIM_TYPES = ["Hail Damage", "Livestock Loss", "Crop Failure", "Equipment Damage", "Fire"]


def generate_claims(n: int = N_CLAIMS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a deterministic, wholly synthetic claims dataset."""

    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    policy_type = rng.choice(["Motor", "Agricultural"], size=n, p=[0.72, 0.28])
    policy_tenure_months = rng.gamma(shape=2.2, scale=14, size=n).clip(1, 240).round().astype(int)
    claimant_age = rng.normal(41, 12, size=n).clip(18, 85).round().astype(int)
    prior_claims_count = rng.poisson(0.6, size=n)
    days_to_report = rng.gamma(shape=1.4, scale=2.3, size=n).clip(0, 60).round().astype(int)
    police_report_filed = rng.random(n) > 0.28
    region = rng.choice(PROVINCES, size=n, p=PROVINCE_WEIGHTS)

    claim_type = np.where(
        policy_type == "Motor",
        rng.choice(MOTOR_CLAIM_TYPES, size=n),
        rng.choice(AG_CLAIM_TYPES, size=n),
    )

    base_amount = np.where(
        policy_type == "Motor",
        rng.lognormal(mean=9.6, sigma=0.55, size=n),
        rng.lognormal(mean=10.3, sigma=0.65, size=n),
    )
    claim_amount = base_amount.round(2)
    high_value_cutoff = np.quantile(claim_amount, 0.85)

    # Latent fraud propensity: logistic combination of risk signals + noise,
    # rescaled so the average matches BASE_FRAUD_RATE. Noise is kept below
    # the combined signal so the problem is learnable but not trivial -
    # an earlier version of this generator made the noise term too large
    # relative to the signal terms, which produced an unlearnable dataset
    # (ROC-AUC ~0.59, recall 0 at the default threshold). Signal strength
    # was increased and noise reduced until the classifier had a real,
    # if imperfect, signal to find.
    z = (
        -3.6
        + 1.3 * (days_to_report > 14)
        + 0.3 * prior_claims_count
        + 1.9 * (~police_report_filed)
        + 0.8 * (policy_tenure_months < 6)
        + 1.0 * (claim_amount > high_value_cutoff)
        + rng.normal(0, 0.3, size=n)
    )
    fraud_prob = 1 / (1 + np.exp(-z))
    fraud_prob = fraud_prob * (BASE_FRAUD_RATE / fraud_prob.mean())
    fraud_prob = np.clip(fraud_prob, 0, 0.97)
    fraud_flag = rng.binomial(1, fraud_prob)

    df = pd.DataFrame({
        "claim_id": [f"MC-{80000 + i}" for i in range(n)],
        "policy_type": policy_type,
        "region": region,
        "claim_type": claim_type,
        "claim_amount": claim_amount,
        "policy_tenure_months": policy_tenure_months,
        "claimant_age": claimant_age,
        "prior_claims_count": prior_claims_count,
        "days_to_report": days_to_report,
        "police_report_filed": police_report_filed,
        "fraud_flag": fraud_flag,
    })
    return df


def write_claims(path: Path = DEFAULT_OUTPUT, *, n: int = N_CLAIMS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate claims and write them to a CSV outside version control."""

    claims = generate_claims(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    claims.to_csv(path, index=False)
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic claims dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-claims", type=int, default=N_CLAIMS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    claims = write_claims(args.output, n=args.n_claims, seed=args.seed)
    LOGGER.info("Generated %s synthetic claims at %s", len(claims), args.output)
    LOGGER.info("Synthetic positive-label rate: %.2f%%", claims["fraud_flag"].mean() * 100)


if __name__ == "__main__":
    main()
