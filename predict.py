#!/usr/bin/env python3
"""Generate IncC transfer-risk probabilities from a prepared feature table."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def main() -> None:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=package_root / "model_bundle_v1.joblib",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="StackingEnsemble")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    bundle = joblib.load(args.model)
    feature_names = list(bundle["feature_names"])
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise KeyError(f"Input is missing {len(missing)} model features: {missing}")
    if args.model_name not in bundle["models"]:
        raise KeyError(f"Unknown model: {args.model_name}")

    X = frame[feature_names].apply(pd.to_numeric, errors="raise").fillna(0).to_numpy()
    probability = bundle["models"][args.model_name].predict_proba(X)[:, 1]
    threshold = float(bundle["thresholds"][args.model_name])
    predicted = (probability >= threshold).astype(int)
    risk_group = np.select(
        [probability < threshold, probability < 0.5],
        ["low", "intermediate"],
        default="high",
    )

    id_column = "plasmid_accession" if "plasmid_accession" in frame.columns else None
    output = pd.DataFrame()
    if id_column:
        output[id_column] = frame[id_column]
    else:
        output["row_id"] = np.arange(1, len(frame) + 1)
    output["predicted_probability"] = probability
    output["fixed_threshold"] = threshold
    output["predicted_class"] = predicted
    output["risk_group"] = risk_group
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
