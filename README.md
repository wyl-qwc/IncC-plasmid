# IncC plasmid transfer-risk prediction model

This repository provides a reproducible workflow for fitting, applying, and evaluating a machine-learning model that estimates the study-defined potential cross-genus transfer risk of IncC plasmids. The workflow begins with prepared accession-level feature matrices. Database acquisition, sequence curation, outcome construction, and raw-sequence feature extraction are described in the associated manuscript and are outside this repository's reproducibility boundary.

## Repository contents

- `Features.zip`: compressed prepared feature matrices and example files.
- `feature_dictionary.csv`: definitions and roles of the released variables.
- `train_model.py`: mutual-information selection of 30 four-mers, model fitting, development-cohort out-of-fold prediction, and bootstrap-smoothed F2 threshold selection.
- `predict.py`: prediction with the fitted model and fixed decision threshold.
- `evaluate.py`: evaluation in the RefSeq cohort, including AUROC, average precision (AP), threshold-based metrics, and stratified-bootstrap 95% confidence intervals.
- `requirements.txt`: tested Python package versions.

The fitted model, `model_bundle_v1.joblib`, is distributed separately as a GitHub Release asset because of its file size.

## Data archive

Extract `Features.zip` in the repository root. It creates a `Features` directory containing:

- `train_features.csv`: PLSDB model-training set (n=804).
- `internal_validation_features.csv`: PLSDB internal-validation set (n=202).
- `external_test_features.csv`: RefSeq evaluation cohort (n=250).
- `example_input.csv`: five-row prediction example containing the 46 required model inputs.
- `example_output.csv`: expected predictions for the example input.

The training matrices contain all candidate 4-mer and 5-mer frequencies required to reproduce mutual-information screening. The final model uses 16 biological predictors and 30 selected 4-mer frequencies. The outcome column is `cross_genus_transfer`.

## Installation

Python 3.11.9 was used for the final analysis.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Download the fitted model

Open the repository's **Releases** page, download `model_bundle_v1.joblib`, and place it in the repository root next to `predict.py`.

Only load a `joblib` model obtained from a trusted source.

## Prediction

```bash
python predict.py --input Features/example_input.csv --output example_predictions.csv
```

The output contains the predicted probability, fixed threshold, binary prediction, and descriptive risk group for each plasmid.

## RefSeq evaluation

```bash
python evaluate.py --output-dir reproduced_results
```

The expected stacking-model estimates are:

- AUROC: 0.9485
- AP: 0.9130
- Sensitivity: 0.8929
- Specificity: 0.8313
- MCC: 0.6950

These estimates use the fixed threshold of 0.2578848561. Confidence intervals are calculated using 2,000 stratified percentile-bootstrap replicates.

## Model refitting

```bash
python train_model.py --output-dir retrained_results
```

The supplied fitted model is the authoritative object for reproducing the manuscript's reported predictions. Refitting is deterministic where supported by the underlying libraries, but small numerical differences can occur across operating systems or library builds.

## Intended use and limitations

The model is intended for reference-library-assisted screening and prioritization of IncC plasmids for subsequent review or experimental validation. It does not directly measure conjugation efficiency and is not intended for clinical or patient-level decision-making.

The outcome depends on Mash similarity, host-genus metadata, database coverage, and the prespecified distance threshold. Closely related IncC lineages can share both sequence features and the inferred outcome. The RefSeq cohort is independent by data source, although its outcome labels were derived within the combined PLSDB-RefSeq similarity network.

## Citation and license

Repository citation details and the final software license will be added before public release.
