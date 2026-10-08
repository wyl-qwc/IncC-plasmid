# IncC database-observed cross-genus occurrence model

This repository contains the compact reproducibility package for the final
LightGBM classifier reported in the manuscript. The model uses 100
mutual-information-selected 4-mer frequencies to score IncC plasmids for the
study-defined outcome: participation in at least one highly similar
cross-genus plasmid pair in the databases examined.

The outcome is a database-derived occurrence label. It is not a measurement of
conjugation efficiency, host range, or clinical risk.

## Contents

- `model/lightgbm100_final.joblib`: LightGBM model fitted to all 988 PLSDB
  plasmids (266 positive and 722 negative).
- `model/model_metadata.json`: model inputs, software versions, label
  definition, and the fixed decision threshold (0.306).
- `data/plsdb_988_4mer_features.csv.gz`: the 256 candidate 4-mer frequencies
  and final labels used for model development.
- `data/refseq_250_4mer_features.csv.gz`: the corresponding RefSeq evaluation
  matrix (84 positive and 166 negative).
- `data/*_biological_features.csv.gz`: harmonized biological covariates used in
  the feature-group ablation; these variables are not inputs to the final model.
- `data/folds_*.csv`: frozen random and Mash-component-blocked five-fold
  assignments used in the manuscript.
- `data/final_selected_100_4mers.csv`: features selected after refitting on the
  full development cohort.
- `scripts/train_final_model.py`: repeat full-cohort feature selection and
  model fitting.
- `scripts/predict.py`: score FASTA sequences or a prepared 4-mer CSV table.
- `scripts/reproduce_cv.py`: reproduce the final LightGBM random and
  lineage-blocked out-of-fold analyses and their bootstrap intervals.
- `scripts/reproduce_threshold.py`: reproduce the nested Youden-index threshold
  analysis used to select the operating cutoff.
- `scripts/construct_labels.py`: reconstruct labels from a frozen pair table,
  host-genus mapping, and final cohort list.
- `scripts/feature_extraction/`: frozen annotation-to-feature rules for CDS,
  AMRFinderPlus, MOB-typer, sequence, topology, and CONJScan variables.

Full pair-level label evidence, reference manifests, mechanistic-analysis source
tables, and per-plasmid manuscript predictions are distributed with the article
as Supplementary Data rather than duplicated here.

The annotation conversion utility can be tested without external databases:

```bash
python -m unittest discover -s scripts/feature_extraction -p "test_rules.py" -v
```

AMRFinderPlus, MOB-suite, and CONJScan are external tools and are not bundled in
this repository. Their versions and frozen output-conversion rules are stated in
`scripts/feature_extraction/README.md`.

## Installation

Python 3.11 was used for the final analysis.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Predict from FASTA

The FASTA workflow counts forward-strand overlapping 4-mers with step size 1.
Windows containing non-ACGT characters are excluded; there is no circular
wraparound or reverse-complement pooling.

```bash
python scripts/predict.py --fasta plasmids.fasta --output predictions.csv
```

FASTA headers are reduced to the first whitespace-delimited token. Each record
must contain at least one valid 4-mer window.

## Predict from a prepared feature table

```bash
python scripts/predict.py --features features.csv --output predictions.csv
```

The input must contain all 100 selected 4-mer columns. An optional
`plasmid_accession` column is retained in the output.

Scores are LightGBM outputs and are not probability-calibrated. The binary call
uses the manuscript operating threshold of 0.306, selected by maximizing the
Youden index within nested random cross-validation. It should not be interpreted
as an absolute biological transfer threshold.

## Refit the released model

```bash
python scripts/train_final_model.py
```

This writes a refitted bundle and metadata to `reproduced/model/`. The supplied
model object is the archival object associated with the manuscript; small
floating-point differences can occur across operating systems or LightGBM
builds.

## Reproduce cross-validation

```bash
python scripts/reproduce_cv.py --bootstrap 2000
python scripts/reproduce_threshold.py
```

The first command uses Mash connected components as bootstrap units throughout:
the primary 0.00123693 components for random CV and the components matching each
lineage-blocked protocol. It uses 2,000 replicates, `numpy.default_rng`, the
frozen base seed 20261004, and the manuscript's protocol-specific Figure 3 seed
offsets (20261104--20261106). Feature selection is repeated inside every
training fold. The second command estimates one Youden cutoff inside each outer
training set and applies it unchanged to the corresponding held-out fold.

Expected discrimination estimates are:

| Validation | AUROC (95% CI) | Average precision (95% CI) |
|---|---:|---:|
| Random five-fold CV | 0.9605 (0.9343--0.9763) | 0.9275 (0.8456--0.9630) |
| Blocked CV, Mash 0.00123693 | 0.7421 (0.6457--0.8140) | 0.4977 (0.3190--0.6428) |
| Blocked CV, Mash 0.0025 | 0.7116 (0.6378--0.7759) | 0.4125 (0.2490--0.5470) |

## Scope and limitations

The model was developed specifically for IncC plasmids. Its label depends on
database sampling, host metadata, and the prespecified Mash threshold. The
RefSeq set is separate by data source, but its outcome was ascertained in the
same frozen PLSDB--RefSeq similarity network. Use the score for computational
screening and prioritization, followed by genomic review or experimental
validation.

