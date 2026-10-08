from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif

ACCESSION = "plasmid_accession"
LABEL = "cross_genus_transfer"
SEED = 42
N_FEATURES = 100
DECISION_THRESHOLD = 0.306
ALL_4MERS = ["".join(x) for x in itertools.product("ACGT", repeat=4)]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_feature_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in [ACCESSION, LABEL, *ALL_4MERS] if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} required columns: {missing[:10]}")
    if frame[ACCESSION].duplicated().any():
        raise ValueError(f"{path} contains duplicate accessions")
    if not set(frame[LABEL].dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL} must be binary")
    return frame


def select_4mers(frame: pd.DataFrame, y: np.ndarray, n: int = N_FEATURES) -> list[str]:
    """Rank by descending MI, then ascending lexical 4-mer for exact ties."""
    scores = mutual_info_classif(
        frame[ALL_4MERS].to_numpy(float), np.asarray(y, dtype=int), random_state=SEED
    )
    ranking = pd.DataFrame({"feature": ALL_4MERS, "score": scores})
    return ranking.sort_values(
        ["score", "feature"], ascending=[False, True], kind="mergesort"
    )["feature"].head(n).tolist()


def new_model() -> LGBMClassifier:
    return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1)


def fit_bundle(frame: pd.DataFrame) -> dict:
    y = frame[LABEL].to_numpy(int)
    features = select_4mers(frame, y)
    model = new_model()
    model.fit(frame[features], y)
    return {
        "model": model,
        "feature_names": features,
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_rule": "nested random-CV Youden index; rounded mean cutoff",
        "label_name": LABEL,
        "label_definition": (
            "database-observed participation in a cross-genus IncC plasmid pair "
            "at Mash distance <= 0.00123693"
        ),
        "random_seed": SEED,
        "training_n": int(len(frame)),
        "training_positive_n": int(y.sum()),
    }


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    required = {"model", "feature_names", "decision_threshold"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    return bundle
