#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import lightgbm
import numpy
import pandas
import sklearn

from common import fit_bundle, read_feature_table, repository_root


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(description="Refit the final LightGBM-100 model")
    parser.add_argument(
        "--data", type=Path, default=root / "data" / "plsdb_988_4mer_features.csv.gz"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "reproduced" / "model")
    args = parser.parse_args()

    frame = read_feature_table(args.data)
    if (len(frame), int(frame.cross_genus_transfer.sum())) != (988, 266):
        raise ValueError("Expected the final 988-plasmid cohort with 266 positives")
    bundle = fit_bundle(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_dir / "lightgbm100_refitted.joblib")

    metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata["software_versions"] = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'lightgbm100_refitted.joblib'}")


if __name__ == "__main__":
    main()
