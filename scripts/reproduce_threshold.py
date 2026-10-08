#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold

from common import ACCESSION, LABEL, new_model, read_feature_table, repository_root, select_4mers


def choose_youden(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    score = tpr - fpr
    finite = np.isfinite(thresholds) & np.isfinite(score)
    best = finite & np.isclose(score, np.max(score[finite]), rtol=0, atol=1e-12)
    return float(np.max(thresholds[best])), float(np.max(score[finite]))


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Reproduce nested Youden threshold selection")
    parser.add_argument("--data", type=Path, default=root / "data" / "plsdb_988_4mer_features.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "reproduced" / "threshold")
    args = parser.parse_args()

    data = read_feature_table(args.data)
    folds = pd.read_csv(root / "data" / "folds_random.csv")
    frame = data.merge(folds, on=ACCESSION, validate="one_to_one")
    manuscript_oof = pd.read_csv(root / "data" / "manuscript_random_oof_predictions.csv")
    frame = frame.merge(
        manuscript_oof[[ACCESSION, "LightGBM_probability"]], on=ACCESSION, validate="one_to_one"
    )
    y = frame[LABEL].to_numpy(int)
    outer_fold = frame["fold"].to_numpy(int)
    probability = frame["LightGBM_probability"].to_numpy(float)
    predicted = np.full(len(frame), -1, dtype=int)
    rows = []

    for fold in sorted(np.unique(outer_fold)):
        outer_train = np.flatnonzero(outer_fold != fold)
        outer_test = np.flatnonzero(outer_fold == fold)
        train = frame.iloc[outer_train].reset_index(drop=True)
        train_y = y[outer_train]
        inner_oof = np.full(len(train), np.nan)
        splitter = StratifiedKFold(5, shuffle=True, random_state=42 + int(fold))
        for inner_train, inner_valid in splitter.split(train, train_y):
            features = select_4mers(train.iloc[inner_train], train_y[inner_train])
            model = new_model()
            model.fit(train.iloc[inner_train][features], train_y[inner_train])
            inner_oof[inner_valid] = model.predict_proba(train.iloc[inner_valid][features])[:, 1]
        threshold, maximum_j = choose_youden(train_y, inner_oof)
        predicted[outer_test] = (probability[outer_test] >= threshold).astype(int)
        rows.append({"outer_fold": int(fold), "threshold": threshold, "maximum_youden_j": maximum_j})
        print(f"Outer fold {fold}: threshold={threshold:.9f}")

    table = pd.DataFrame(rows)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    summary = {
        "threshold_mean": float(table.threshold.mean()),
        "threshold_mean_rounded_for_final_model": 0.306,
        "threshold_median": float(table.threshold.median()),
        "threshold_min": float(table.threshold.min()),
        "threshold_max": float(table.threshold.max()),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "fold_thresholds.csv", index=False)
    (args.output_dir / "threshold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
