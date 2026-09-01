"""Final project gate used locally and by GitHub Actions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.validate_outputs import validate

ROOT = Path(__file__).resolve().parent
REQUIRED_PATHS = {
    ".github/workflows/ci.yml",
    "src/features.py",
    "src/models.py",
    "src/metrics.py",
    "src/evaluate.py",
    "src/train.py",
    "tests/test_data.py",
    "tests/test_metrics.py",
    "tests/test_pipeline.py",
    "tests/test_privacy.py",
}
REQUIRED_README_LANGUAGE = {
    "synthetic data only",
    "uncalibrated model score",
    "human-reviewed priority queue",
    "leakage-safe pipeline",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    missing = sorted(path for path in REQUIRED_PATHS if not (ROOT / path).is_file())
    if missing:
        raise ValueError(f"Missing project files: {missing}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    missing_language = sorted(text for text in REQUIRED_README_LANGUAGE if text not in readme)
    if missing_language:
        raise ValueError(f"README is missing required language: {missing_language}")
    validate(args.results_dir)
    manifest = json.loads((args.results_dir / "run-manifest.json").read_text(encoding="utf-8"))
    if not manifest["fast_mode"]:
        summary = json.loads(
            (args.results_dir / "holdout-summary.json").read_text(encoding="utf-8")
        )
        with (args.results_dir / "operating-points.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            top_ten = next(
                row for row in csv.DictReader(handle) if float(row["capacity"]) == 0.10
            )
        expected_readme_values = {
            f"{summary['roc_auc']:.3f} roc-auc",
            f"{summary['pr_auc']:.3f} pr-auc",
            f"{float(top_ten['precision']):.2%} precision",
            f"{float(top_ten['recall']):.2%} recall",
            f"{float(top_ten['lift']):.2f}x lift",
        }
        missing_values = sorted(value for value in expected_readme_values if value not in readme)
        if missing_values:
            raise ValueError(f"README metrics do not match tracked results: {missing_values}")
    print("PASS: project structure, responsible-use language and generated evidence validated")


if __name__ == "__main__":
    main()
