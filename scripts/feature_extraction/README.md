# Harmonized feature collection rules

This directory freezes the feature definitions used for both the PLSDB development
cohort and the RefSeq external cohort. It separates raw input, external annotation
tools, and conversion of tool output into model variables.

The exact regular expressions and structured-field rules are in `rules.json`.
`match_annotations.py` replays IS, AMRFinderPlus and MOB-typer output conversion using
only the Python standard library. `FEATURE_COLLECTION.md` gives the complete
collection specification used for the released analysis cohorts.

The current harmonized annotation tools are:

- AMRFinderPlus 4.2.7, database 2026-05-15.1, nucleotide mode with `--plus`;
  only records whose `Type` is exactly `AMR` contribute to ARG features.
- MOB-suite 3.1.9 `mob_typer`, default settings and four threads.
- MacSyFinder 2.1.6 with CONJScan 2.1.0, model `CONJScan/Plasmids` and
  `--db-type ordered_replicon`.

Example commands:

```bash
python scripts/feature_extraction/match_annotations.py --mode cds --input scripts/feature_extraction/examples/cds.csv --output cds.json
python scripts/feature_extraction/match_annotations.py --mode amrfinder --delimiter tab --input scripts/feature_extraction/examples/amrfinder.tsv --output amr.json
python scripts/feature_extraction/match_annotations.py --mode mob --delimiter tab --input scripts/feature_extraction/examples/mobtyper.tsv --output mob.json
python -m unittest discover -s scripts/feature_extraction -p "test_rules.py" -v
```

CDS input requires `plasmid_accession`, `gene`, and `product`. AMRFinderPlus input
requires the native `Type`, `Subclass`, `Element name`, and `Class` fields, plus
`plasmid_accession` and `plasmid_length` joined onto every row. MOB-typer input
requires `plasmid_accession`, `orit_type(s)`, `relaxase_type(s)`, `mpf_type`, and
`predicted_mobility`.

The utility converts completed external-tool outputs and does not run the tools
or infer a biological zero when an annotation run is absent. Labels and model
fitting are handled by separate scripts.
