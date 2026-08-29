"""
Trains and evaluates an XGBoost fraud classifier on the synthetic claims
dataset produced by generate_data.py.

Portfolio/demo pipeline - synthetic data only, not connected to any
production or employer system. SMOTE is applied to the training fold
only, after the train/test split, to avoid leaking synthetic minority
samples into the evaluation set.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from generate_data import generate_claims

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def build_features(df):
    cat_cols = ["policy_type", "region", "claim_type"]
    num_cols = ["claim_amount", "policy_tenure_months", "claimant_age",
                "prior_claims_count", "days_to_report"]

    X_num = df[num_cols].copy()
    X_num["police_report_filed"] = df["police_report_filed"].astype(int)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[cat_cols])
    feature_names = num_cols + ["police_report_filed"] + list(encoder.get_feature_names_out(cat_cols))

    X = np.hstack([X_num.values, X_cat])
    y = df["fraud_flag"].to_numpy()
    return X, y, feature_names


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_at_flag_rate(y_true, y_prob, flag_rate):
    """Flag the top `flag_rate` fraction of claims by predicted risk.

    This is how SIU/claims-review capacity is actually allocated in
    practice - a fixed probability threshold (e.g. 0.5) is nearly
    meaningless when the positive class is ~5% of the data, since
    predicted probabilities for a rare event rarely approach 0.5 even
    for genuine positives. Flagging by percentile instead ties the
    model's output to a resourcing decision ("we can review the top
    10% of claims") rather than an arbitrary probability cutoff.
    """
    n_flag = max(1, int(round(len(y_prob) * flag_rate)))
    flagged_idx = np.argsort(y_prob)[::-1][:n_flag]
    y_pred = np.zeros_like(y_true)
    y_pred[flagged_idx] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "flag_rate": flag_rate,
        "n_flagged": int(n_flag),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    df = generate_claims()
    X, y, feature_names = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_res, y_train_res)

    # Time inference the way it would run in production: one claim at a time.
    single_times = []
    for i in range(min(500, len(X_test))):
        row = X_test[i : i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        single_times.append(time.perf_counter() - t0)
    avg_inference_ms = float(np.mean(single_times) * 1000)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Illustrates the naive-threshold trap: a fixed 0.5 cutoff gives ~95%
    # "accuracy" but 0% recall, because predicted probabilities for a rare
    # event rarely cross 0.5 even for true positives. Kept deliberately as
    # a documented caution, not hidden.
    naive_threshold_trap = evaluate_at_threshold(y_test, y_prob, 0.5)

    # The actual recommended framing: flag a review capacity, not a probability.
    flag_rate_sweep = [evaluate_at_flag_rate(y_test, y_prob, r) for r in (0.05, 0.10, 0.15, 0.20)]
    recommended_operating_point = evaluate_at_flag_rate(y_test, y_prob, 0.10)

    metrics = {
        "n_claims_total": int(len(df)),
        "n_test_claims": int(len(X_test)),
        "test_fraud_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "avg_inference_ms_per_claim": avg_inference_ms,
        "naive_threshold_trap_0.5": naive_threshold_trap,
        "flag_rate_sweep": flag_rate_sweep,
        "recommended_operating_point_top_10pct": recommended_operating_point,
        "top_features_by_gain": get_top_features(model, feature_names),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def get_top_features(model, feature_names, top_n=8):
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    named = {feature_names[int(k[1:])]: v for k, v in scores.items()}
    ranked = sorted(named.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"feature": f, "gain": round(g, 2)} for f, g in ranked]


if __name__ == "__main__":
    main()
