# End-to-end feature collection specification

The same accession-versioned nucleotide sequence is the unit of analysis in both
cohorts. Accessions must not be silently version-stripped when selecting inputs.

| Feature(s) | Frozen input and calculation |
|---|---|
| `plasmid_length` | Number of characters in the upper-case FASTA sequence after removing whitespace. |
| `gc_content` | `(G + C) / sequence length`; ambiguous bases remain in the denominator. |
| 4-mer frequencies | Forward strand, overlapping windows, no circular wrap and no reverse-complement pooling. A window is counted only if every character is A/C/G/T. Divide by the number of valid windows. Ambiguous bases stay in position and invalidate overlapping windows. |
| `orf_count` | Target definition: number of GenBank `CDS` features. A missing annotation record is missing data, not zero. |
| `coding_density` | Extract every part of every GenBank CDS location as a 0-based half-open interval, take the interval-union length, divide by plasmid length and round to four decimals. Joined or origin-spanning CDS are represented by their parts. |
| `IS_count` | Count CDS annotation rows whose `product` matches at least one pattern in `rules.json`; a row counts once. |
| Five ARG variables | Run AMRFinderPlus 4.2.7/database 2026-05-15.1 in nucleotide mode with `--plus`, retain `Type == AMR`, then apply the structured rules in `rules.json`. |
| Four MOB variables | Run MOB-suite 3.1.9 `mob_typer` with default settings and four threads, then apply the field rules in `rules.json`. |
| `tra_cluster_completeness` | Translate annotated CDS, run MacSyFinder 2.1.6/CONJScan 2.1.0 with `CONJScan/Plasmids` and `ordered_replicon`, drop duplicate `sys_id`, and take maximum `sys_wholeness`, rounded to three decimals. Zero is allowed only after a successful run with no detected system. |
| `topology_binary` | `1` when the accession-versioned GenBank topology is `circular`, otherwise `0`. No topology annotations were missing in the final 988-plasmid PLSDB development cohort or the 250-plasmid RefSeq evaluation cohort. |

An earlier extraction audit covered 1,252 records with available GenBank files.
The released analysis matrices contain the final 1,238 observations
(988 PLSDB plus 250 RefSeq) after cohort filtering; the audit count is not an
analysis sample size.

External-tool versions, database versions, command lines, sequence checksums,
and per-accession completion status were retained with the analysis records.
Failed or absent annotation runs were treated as missing until resolved. Four
PLSDB accessions without recoverable CDS annotation (`PP622803.1`, `AP018579.2`,
`LT904892.1`, and `OL332704.1`) were excluded before the final development
cohort was formed and were not filled with zero.
