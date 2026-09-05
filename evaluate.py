#!/usr/bin/env python3
"""Evaluate the saved model with AUROC, AP and fixed-threshold metrics."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


LABEL = "cross_genus_transfer"


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "F2": fbeta_score(y, pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
    }


def stratified_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        index = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        current = metrics(y[index], p[index], threshold)
        for name, value in current.items():
            if name not in {"TN", "FP", "FN", "TP"}:
                samples.setdefault(name, []).append(value)
    return {
        name: tuple(np.percentile(values, [2.5, 97.5]).tolist())
        for name, values in samples.items()
    }


def main() -> None:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=package_root / "Features" / "external_test_features.csv"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=package_root / "model_bundle_v1.joblib",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="StackingEnsemble")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if LABEL not in frame.columns:
        raise KeyError(f"Evaluation input must contain {LABEL}")
    bundle = joblib.load(args.model)
    feature_names = list(bundle["feature_names"])
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise KeyError(f"Evaluation input is missing model features: {missing}")

    y = frame[LABEL].to_numpy(dtype=int)
    X = frame[feature_names].apply(pd.to_numeric, errors="raise").fillna(0).to_numpy()
    probability = bundle["models"][args.model_name].predict_proba(X)[:, 1]
    threshold = float(bundle["thresholds"][args.model_name])
    estimates = metrics(y, probability, threshold)
    intervals = stratified_bootstrap(y, probability, threshold, args.bootstrap, args.seed)

    rows = []
    for name, estimate in estimates.items():
        low, high = intervals.get(name, (np.nan, np.nan))
        rows.append(
            {
                "Metric": name,
                "Estimate": estimate,
                "CI_low": low,
                "CI_high": high,
                "CI_method": "stratified percentile bootstrap" if name in intervals else "not applicable",
                "Bootstrap_repeats": args.bootstrap if name in intervals else np.nan,
            }
        )
    predictions = pd.DataFrame(
        {
            "plasmid_accession": frame.get(
                "plasmid_accession", pd.Series(np.arange(1, len(frame) + 1))
            ),
            LABEL: y,
            "predicted_probability": probability,
            "fixed_threshold": threshold,
            "predicted_class": (probability >= threshold).astype(int),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics_with_95CI.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
