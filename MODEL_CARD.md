# Model card

## Model

LightGBM with software-default classifier hyperparameters, a fixed random seed
of 42, and 100 4-mer frequency features selected by mutual information.

## Training data

The final development cohort contains 988 PLSDB IncC plasmids: 266 positive and
722 negative under the operational outcome definition. The external evaluation
cohort contains 250 nonduplicate complete RefSeq IncC plasmids: 84 positive and
166 negative.

## Outcome

A plasmid is positive when it participates in at least one pair at Mash distance
less than or equal to 0.00123693 whose two resolved host genera differ. A
negative label means that no qualifying pair was observed in the frozen data
sources. The outcome does not directly measure conjugation.

## Output

The model returns an uncalibrated score and a binary call at the fixed threshold
0.306. The threshold is the rounded mean of the five cutoffs selected by the
Youden index within the outer-training folds of nested random five-fold CV.

## Appropriate use

Computational screening and prioritization of IncC plasmids for subsequent
genomic or experimental review.

## Limitations

- Closely related plasmids can share both sequence composition and labels.
- Performance is lower when Mash-connected components are blocked between
  training and test folds.
- The outcome depends on database coverage and host-genus annotation.
- The RefSeq evaluation has separate records but shared network-based outcome
  ascertainment.
- The model is not intended for clinical decisions or quantitative prediction
  of conjugation efficiency.
