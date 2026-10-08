#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import ACCESSION, ALL_4MERS, load_bundle, repository_root


def read_fasta(path: Path):
    name = None
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence).upper()
                name = line[1:].split()[0]
                if not name:
                    raise ValueError("FASTA record has an empty identifier")
                sequence = []
            else:
                if name is None:
                    raise ValueError("Sequence data appeared before the first FASTA header")
                sequence.append(line)
    if name is not None:
        yield name, "".join(sequence).upper()


def fasta_features(path: Path) -> pd.DataFrame:
    rows = []
    for accession, sequence in read_fasta(path):
        valid = [
            sequence[i : i + 4]
            for i in range(max(0, len(sequence) - 3))
            if set(sequence[i : i + 4]) <= set("ACGT")
        ]
        if not valid:
            raise ValueError(f"{accession} has no valid A/C/G/T 4-mer window")
        counts = Counter(valid)
        row = {ACCESSION: accession}
        row.update({kmer: counts.get(kmer, 0) / len(valid) for kmer in ALL_4MERS})
        rows.append(row)
    if not rows:
        raise ValueError("No FASTA records were found")
    return pd.DataFrame(rows)


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Score IncC plasmid sequences")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fasta", type=Path)
    inputs.add_argument("--features", type=Path)
    parser.add_argument("--model", type=Path, default=root / "model" / "lightgbm100_final.joblib")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = fasta_features(args.fasta) if args.fasta else pd.read_csv(args.features)
    bundle = load_bundle(args.model)
    selected = list(bundle["feature_names"])
    missing = [feature for feature in selected if feature not in frame]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} selected 4-mers: {missing[:10]}")
    X = frame[selected].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Selected 4-mer columns contain missing values")
    score = bundle["model"].predict_proba(X)[:, 1]
    cutoff = float(bundle["decision_threshold"])
    output = pd.DataFrame({
        ACCESSION: frame[ACCESSION] if ACCESSION in frame else np.arange(1, len(frame) + 1),
        "model_score": score,
        "decision_threshold": cutoff,
        "predicted_label": (score >= cutoff).astype(int),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
