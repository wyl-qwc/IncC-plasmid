#!/usr/bin/env python3
"""Train the IncC cross-genus transfer-risk model from prepared feature tables.

The script mirrors the final analysis design while using the standardized names
``has_MPF`` and ``cross_genus_transfer``. Raw-sequence feature extraction is
outside this release boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


SEED = 42
N_FOLDS = 5
LABEL = "cross_genus_transfer"
ACCESSION = "plasmid_accession"
BEST_K4 = 30
THRESHOLD_GRID = np.linspace(0.02, 0.85, 800)

BASIC_FEATURES = [
    "plasmid_length",
    "gc_content",
    "orf_count",
    "has_oriT",
    "has_relaxase",
    "has_MPF",
    "total_arg_count",
    "has_carbapenemase",
    "has_ESBL",
    "coding_density",
    "IS_count",
    "ARG_class_count",
    "ARG_density",
    "topology_binary",
    "mobility",
    "tra_cluster_completeness",
]

TUNED_PARAMS = {
    "RandomForest": {
        "max_depth": 6,
        "max_features": 0.5,
        "min_samples_leaf": 5,
        "n_estimators": 800,
    },
    "ExtraTrees": {
        "max_depth": 6,
        "max_features": 0.8,
        "min_samples_leaf": 5,
        "n_estimators": 800,
    },
    "LightGBM": {
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": 400,
        "num_leaves": 31,
    },
    "XGBoost": {
        "learning_rate": 0.03,
        "max_depth": 8,
        "n_estimators": 400,
        "subsample": 0.85,
    },
    "HistGradientBoosting": {
        "l2_regularization": 0.0,
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 400,
    },
}


def base_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            **TUNED_PARAMS["RandomForest"], random_state=SEED, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            **TUNED_PARAMS["ExtraTrees"], random_state=SEED, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            **TUNED_PARAMS["LightGBM"], random_state=SEED, verbose=-1
        ),
        "XGBoost": XGBClassifier(
            **TUNED_PARAMS["XGBoost"],
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            **TUNED_PARAMS["HistGradientBoosting"], random_state=SEED
        ),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("scale", RobustScaler()),
            ("resample", "passthrough"),
            ("clf", model),
        ]
    )


def make_stacking() -> StackingClassifier:
    estimators = [(name, make_pipeline(model)) for name, model in base_models().items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=N_FOLDS,
        n_jobs=1,
    )


def select_4mers(train: pd.DataFrame) -> list[str]:
    candidates = sorted(
        column
        for column in train.columns
        if len(column) == 4 and set(column).issubset({"A", "C", "G", "T"})
    )
    if len(candidates) != 256:
        raise ValueError(f"Expected 256 four-mer columns, found {len(candidates)}")
    scores = mutual_info_classif(
        train[candidates].to_numpy(), train[LABEL].to_numpy(), random_state=SEED
    )
    return [candidates[index] for index in np.argsort(scores)[-BEST_K4:]]


def validate_table(frame: pd.DataFrame, name: str) -> None:
    required = {ACCESSION, LABEL, *BASIC_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")
    if not frame[LABEL].isin([0, 1]).all():
        raise ValueError(f"{name}: {LABEL} must be binary")


def f2_threshold(y_true: np.ndarray, probabilities: np.ndarray, n_bootstrap: int) -> float:
    """Select the fixed threshold from the mean bootstrap F2 curve."""
    rng = np.random.RandomState(SEED)
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    mean_scores = np.zeros_like(THRESHOLD_GRID, dtype=float)
    for _ in range(n_bootstrap):
        index = rng.choice(len(y_true), size=len(y_true), replace=True)
        y = y_true[index][:, None]
        pred = probabilities[index][:, None] >= THRESHOLD_GRID[None, :]
        tp = np.sum(pred & (y == 1), axis=0)
        fp = np.sum(pred & (y == 0), axis=0)
        fn = np.sum((~pred) & (y == 1), axis=0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        score = np.divide(
            5 * precision * recall,
            4 * precision + recall,
            out=np.zeros_like(precision),
            where=(4 * precision + recall) > 0,
        )
        mean_scores += score
    mean_scores /= n_bootstrap
    return float(THRESHOLD_GRID[int(np.argmax(mean_scores))])


def oof_probabilities(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    names = [*base_models(), "StackingEnsemble"]
    oof = {name: np.zeros(len(y), dtype=float) for name in names}
    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        for name, model in base_models().items():
            calibrated = CalibratedClassifierCV(
                make_pipeline(model), method="isotonic", cv=N_FOLDS
            )
            calibrated.fit(X_train, y_train)
            oof[name][test_index] = calibrated.predict_proba(X_test)[:, 1]
        stacking = make_stacking()
        stacking.fit(X_train, y_train)
        oof["StackingEnsemble"][test_index] = stacking.predict_proba(X_test)[:, 1]
    return oof


def fit_final_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    fitted = {}
    for name, model in base_models().items():
        calibrated = CalibratedClassifierCV(
            make_pipeline(model), method="isotonic", cv=N_FOLDS
        )
        calibrated.fit(X_train, y_train)
        fitted[name] = calibrated
    stacking = make_stacking()
    stacking.fit(X_train, y_train)
    fitted["StackingEnsemble"] = stacking
    return fitted


def metric_row(name: str, y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": threshold,
        "AUROC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y, pred, zero_division=0),
        "F2": fbeta_score(y, pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=package_root / "Features" / "train_features.csv")
    parser.add_argument(
        "--internal-validation",
        type=Path,
        default=package_root / "Features" / "internal_validation_features.csv",
    )
    parser.add_argument(
        "--external", type=Path, default=package_root / "Features" / "external_test_features.csv"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    internal = pd.read_csv(args.internal_validation)
    external = pd.read_csv(args.external)
    for name, frame in (("training", train), ("internal validation", internal), ("external", external)):
        validate_table(frame, name)

    selected_4mers = select_4mers(train)
    feature_names = BASIC_FEATURES + selected_4mers
    X_train = train[feature_names].fillna(0).to_numpy()
    y_train = train[LABEL].to_numpy(dtype=int)
    development = pd.concat([train, internal], ignore_index=True)
    X_development = development[feature_names].fillna(0).to_numpy()
    y_development = development[LABEL].to_numpy(dtype=int)

    oof = oof_probabilities(X_development, y_development)
    thresholds = {
        name: f2_threshold(y_development, probabilities, args.threshold_bootstrap)
        for name, probabilities in oof.items()
    }
    fitted_models = fit_final_models(X_train, y_train)

    bundle = {
        "models": fitted_models,
        "thresholds": thresholds,
        "final_threshold_strategy": "f2",
        "feature_names": feature_names,
        "basic_features": BASIC_FEATURES,
        "selected_kmer_features": selected_4mers,
        "seed": SEED,
    }
    joblib.dump(bundle, args.output_dir / "model_bundle_retrained.joblib")

    X_external = external[feature_names].fillna(0).to_numpy()
    y_external = external[LABEL].to_numpy(dtype=int)
    metrics = []
    predictions = pd.DataFrame({ACCESSION: external[ACCESSION], LABEL: y_external})
    for name, model in fitted_models.items():
        probability = model.predict_proba(X_external)[:, 1]
        metrics.append(metric_row(name, y_external, probability, thresholds[name]))
        predictions[f"{name}_probability"] = probability
    pd.DataFrame(metrics).to_csv(args.output_dir / "external_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "external_predictions.csv", index=False)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "training_n": len(train),
                "internal_validation_n": len(internal),
                "external_n": len(external),
                "selected_4mers": selected_4mers,
                "thresholds": thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
