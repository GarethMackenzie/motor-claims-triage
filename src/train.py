"""Reproducible model development and untouched-holdout evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .config import DEFAULT_CONFIG
from .evaluate import cross_validate_pipeline, evaluate_holdout
from .features import feature_columns, make_model_pipeline, split_features_target
from .generate_data import generate_claims
from .models import candidate_estimators
from .plots import (
    plot_cumulative_gains,
    plot_lift,
    plot_model_comparison,
    plot_permutation_importance,
    plot_precision_recall,
    plot_roc,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_experiment(
    *,
    output_dir: Path = DEFAULT_RESULTS,
    fast: bool = False,
    n_claims: int | None = None,
) -> dict[str, Any]:
    """Run development CV, final selection and one holdout evaluation."""

    config = DEFAULT_CONFIG
    sample_size = n_claims or (2_000 if fast else config.n_claims)
    cv_folds = 3 if fast else config.cv_folds
    permutation_repeats = 3 if fast else config.permutation_repeats
    data = generate_claims(n=sample_size, seed=config.random_state)
    features, target = split_features_target(data)
    development_x, holdout_x, development_y, holdout_y = train_test_split(
        features,
        target,
        test_size=config.holdout_fraction,
        stratify=target,
        random_state=config.random_state,
    )
    positive_weight = float((development_y == 0).sum() / (development_y == 1).sum())
    estimators = candidate_estimators(
        random_state=config.random_state,
        positive_weight=positive_weight,
        fast=fast,
    )

    comparison_rows: list[dict[str, float | str]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    for model_name, estimator in estimators.items():
        LOGGER.info("Cross-validating %s", model_name)
        pipeline = make_model_pipeline(estimator)
        summary, records = cross_validate_pipeline(
            model_name,
            pipeline,
            development_x,
            development_y,
            folds=cv_folds,
            random_state=config.random_state,
        )
        comparison_rows.append(summary)
        fold_rows.extend(records)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["PR-AUC", "Lift@10%"], ascending=False
    )
    selected_name = str(comparison.iloc[0]["Model"])
    LOGGER.info("Selected %s using development cross-validated PR-AUC", selected_name)

    governance_rows: list[dict[str, float | str | bool]] = []
    for include_age in (True, False):
        governed_x, governed_y = split_features_target(data, include_age=include_age)
        governed_development_x = governed_x.loc[development_x.index]
        governed_pipeline = make_model_pipeline(
            estimators[selected_name], include_age=include_age
        )
        summary, _ = cross_validate_pipeline(
            selected_name,
            governed_pipeline,
            governed_development_x,
            governed_y.loc[development_y.index],
            folds=cv_folds,
            random_state=config.random_state,
        )
        governance_rows.append({"Includes claimant age": include_age, **summary})

    governance = pd.DataFrame(governance_rows)
    with_age = governance.loc[governance["Includes claimant age"]].iloc[0]
    without_age = governance.loc[~governance["Includes claimant age"]].iloc[0]
    pr_auc_loss = float(with_age["PR-AUC"] - without_age["PR-AUC"])
    lift_loss = float(with_age["Lift@10%"] - without_age["Lift@10%"])
    final_include_age = not (pr_auc_loss <= 0.01 and lift_loss <= 0.10)
    LOGGER.info(
        "Final model includes claimant age: %s (development-only governance decision)",
        final_include_age,
    )

    final_features, _ = split_features_target(data, include_age=final_include_age)
    final_development_x = final_features.loc[development_x.index]
    final_holdout_x = final_features.loc[holdout_x.index]
    selected_pipeline = make_model_pipeline(
        estimators[selected_name], include_age=final_include_age
    )
    selected_pipeline.fit(final_development_x, development_y)
    holdout_scores = selected_pipeline.predict_proba(final_holdout_x)[:, 1]
    holdout_summary, operating_points = evaluate_holdout(
        holdout_y,
        holdout_scores,
        config.capacities,
    )
    holdout_summary.update(
        {
            "selected_model": selected_name,
            "selection_rule": "Highest mean PR-AUC on stratified development folds; Lift@10% breaks ties.",
            "includes_claimant_age": final_include_age,
            "age_governance_rule": "Exclude age when development mean PR-AUC falls by no more than 0.01 and Lift@10% by no more than 0.10.",
            "score_language": "Uncalibrated model score used for relative review priority.",
        }
    )

    importance_result = permutation_importance(
        selected_pipeline,
        final_holdout_x,
        holdout_y,
        scoring="average_precision",
        n_repeats=permutation_repeats,
        random_state=config.random_state,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_columns(include_age=final_include_age),
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model-comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "cross-validation-folds.csv", index=False)
    operating_frame = pd.DataFrame(operating_points)
    operating_frame.to_csv(output_dir / "operating-points.csv", index=False)
    governance.to_csv(output_dir / "governance-age-comparison.csv", index=False)
    importance.to_csv(output_dir / "permutation-importance.csv", index=False)
    _write_json(output_dir / "holdout-summary.json", holdout_summary)
    manifest = {
        "synthetic_data_only": True,
        "random_state": config.random_state,
        "n_claims": sample_size,
        "development_claims": int(len(development_x)),
        "holdout_claims": int(len(holdout_x)),
        "holdout_fraction": config.holdout_fraction,
        "cv_folds": cv_folds,
        "selected_model": selected_name,
        "selection_metric": "development cross-validated PR-AUC",
        "final_includes_claimant_age": final_include_age,
        "age_decision_used_holdout": False,
        "smote_used": False,
        "holdout_used_for_model_selection": False,
        "fast_mode": fast,
    }
    _write_json(output_dir / "run-manifest.json", manifest)

    target_array = holdout_y.to_numpy()
    plot_roc(target_array, holdout_scores, output_dir / "roc-curve.png")
    plot_precision_recall(target_array, holdout_scores, output_dir / "precision-recall-curve.png")
    plot_cumulative_gains(target_array, holdout_scores, output_dir / "cumulative-gains.png")
    plot_lift(target_array, holdout_scores, output_dir / "lift-curve.png")
    plot_model_comparison(comparison, output_dir / "model-comparison.png")
    plot_permutation_importance(importance, output_dir / "permutation-importance.png")
    return {
        "comparison": comparison,
        "holdout": holdout_summary,
        "operating_points": operating_frame,
        "governance": governance,
        "importance": importance,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate the risk-triage case study.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fast", action="store_true", help="Use a smaller CI smoke-test run.")
    parser.add_argument("--n-claims", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_experiment(output_dir=args.output_dir, fast=args.fast, n_claims=args.n_claims)
    LOGGER.info("Final holdout summary: %s", json.dumps(result["holdout"], indent=2))


if __name__ == "__main__":
    main()
