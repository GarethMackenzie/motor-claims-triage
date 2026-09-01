"""Fail fast when generated model evidence is missing or malformed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

EXPECTED_FILES = {
    "model-comparison.csv", "cross-validation-folds.csv", "operating-points.csv",
    "governance-age-comparison.csv", "permutation-importance.csv",
    "holdout-summary.json", "run-manifest.json", "roc-curve.png",
    "precision-recall-curve.png", "cumulative-gains.png", "lift-curve.png",
    "model-comparison.png", "permutation-importance.png",
}


def validate(output_dir: Path) -> None:
    missing = sorted(name for name in EXPECTED_FILES if not (output_dir / name).is_file())
    if missing:
        raise ValueError(f"Missing output files: {missing}")
    empty = sorted(name for name in EXPECTED_FILES if (output_dir / name).stat().st_size == 0)
    if empty:
        raise ValueError(f"Empty output files: {empty}")

    comparison = pd.read_csv(output_dir / "model-comparison.csv")
    expected_models = {
        "Dummy baseline", "Logistic regression", "Random forest", "XGBoost",
        "Weighted XGBoost",
    }
    if set(comparison["Model"]) != expected_models:
        raise ValueError("Model comparison does not contain the governed benchmark set")

    operating = pd.read_csv(output_dir / "operating-points.csv")
    if set(operating["capacity"].round(2)) != {0.05, 0.10, 0.15, 0.20}:
        raise ValueError("Operating points must cover 5%, 10%, 15% and 20%")
    for column in ["precision", "recall", "false_positive_rate", "lift"]:
        if not operating[column].map(math.isfinite).all():
            raise ValueError(f"Non-finite values found in {column}")

    summary = json.loads((output_dir / "holdout-summary.json").read_text(encoding="utf-8"))
    for key in ["selected_model", "roc_auc", "pr_auc", "n_holdout", "positive_labels"]:
        if key not in summary:
            raise ValueError(f"Holdout summary is missing {key}")
    manifest = json.loads((output_dir / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("holdout_used_for_model_selection") is not False:
        raise ValueError("Holdout isolation is not documented")
    if manifest.get("synthetic_data_only") is not True:
        raise ValueError("Synthetic-data disclosure is not documented")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    validate(args.output_dir)
    print(f"PASS: validated {len(EXPECTED_FILES)} output files in {args.output_dir}")


if __name__ == "__main__":
    main()
