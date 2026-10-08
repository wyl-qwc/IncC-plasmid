#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import ACCESSION, LABEL, new_model, read_feature_table, repository_root, select_4mers


PROTOCOLS = {
    "random": "folds_random.csv",
    "blocked_0.00123693": "folds_blocked_0p00123693.csv",
    "blocked_0.0025": "folds_blocked_0p0025.csv",
}
BOOTSTRAP_SEED = 20261004


def bootstrap_ci(y, probability, groups, repetitions, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    aurocs, aps = [], []
    while len(aurocs) < repetitions:
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([lookup[group] for group in sampled])
        if np.unique(y[index]).size < 2:
            continue
        aurocs.append(roc_auc_score(y[index], probability[index]))
        aps.append(average_precision_score(y[index], probability[index]))
    return np.percentile(aurocs, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Reproduce final LightGBM-100 OOF analyses")
    parser.add_argument("--data", type=Path, default=root / "data" / "plsdb_988_4mer_features.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "reproduced" / "cross_validation")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    data = read_feature_table(args.data)
    if (len(data), int(data[LABEL].sum())) != (988, 266):
        raise ValueError("Expected 988 plasmids and 266 positives")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    primary_components = pd.read_csv(
        root / "data" / "folds_blocked_0p00123693.csv",
        usecols=[ACCESSION, "mash_component"],
    )

    for protocol_index, (protocol, filename) in enumerate(PROTOCOLS.items()):
        assignments = pd.read_csv(root / "data" / filename)
        frame = data.merge(assignments, on=ACCESSION, validate="one_to_one")
        if len(frame) != len(data):
            raise ValueError(f"Incomplete fold assignment for {protocol}")
        y = frame[LABEL].to_numpy(int)
        probability = np.full(len(frame), np.nan)
        selection_rows = []
        for fold in sorted(frame["fold"].unique()):
            train = frame["fold"].to_numpy() != fold
            test = ~train
            features = select_4mers(frame.loc[train], y[train])
            model = new_model()
            model.fit(frame.loc[train, features], y[train])
            probability[test] = model.predict_proba(frame.loc[test, features])[:, 1]
            selection_rows.extend(
                {"protocol": protocol, "fold": int(fold), "rank": rank, "feature": feature}
                for rank, feature in enumerate(features, start=1)
            )
        if np.isnan(probability).any():
            raise RuntimeError(f"Missing OOF predictions for {protocol}")
        evaluation = frame[[ACCESSION, LABEL, "fold"]].copy()
        evaluation["LightGBM_probability"] = probability
        if protocol == "random":
            evaluation = evaluation.merge(
                primary_components, on=ACCESSION, validate="one_to_one"
            )
            component_threshold = "0.00123693"
        else:
            evaluation["mash_component"] = frame["mash_component"].to_numpy()
            component_threshold = "0.00123693" if protocol.endswith("0.00123693") else "0.0025"
        # The manuscript CI calculation sorts accessions before component
        # resampling and uses NumPy's Generator with the frozen seed sequence.
        evaluation = evaluation.sort_values(ACCESSION).reset_index(drop=True)
        y_evaluation = evaluation[LABEL].to_numpy(int)
        probability_evaluation = evaluation["LightGBM_probability"].to_numpy(float)
        groups = evaluation["mash_component"].to_numpy()
        auc_ci, ap_ci = bootstrap_ci(
            y_evaluation,
            probability_evaluation,
            groups,
            args.bootstrap,
            BOOTSTRAP_SEED + 100 + protocol_index,
        )
        metric_rows.append({
            "validation": protocol,
            "n": len(y),
            "positive_n": int(y.sum()),
            "AUROC": roc_auc_score(y, probability),
            "AUROC_CI_low": auc_ci[0],
            "AUROC_CI_high": auc_ci[1],
            "average_precision": average_precision_score(y, probability),
            "AP_CI_low": ap_ci[0],
            "AP_CI_high": ap_ci[1],
            "bootstrap_unit": f"Mash connected component ({component_threshold})",
            "bootstrap_repetitions": args.bootstrap,
        })
        evaluation.to_csv(args.output_dir / f"{protocol}_oof_predictions.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(
            args.output_dir / f"{protocol}_selected_4mers.csv", index=False
        )
        print(f"Completed {protocol}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
