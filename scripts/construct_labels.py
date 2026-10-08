#!/usr/bin/env python3
"""Construct the operational label from frozen pair and host tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

UNRESOLVED = {"", "nan", "none", "uncultured", "enterobacteriaceae"}


def read_pairs(path: Path, separator: str) -> pd.DataFrame:
    pairs = pd.read_csv(path, sep=separator)
    if not {"id_a", "id_b", "distance"}.issubset(pairs.columns):
        pairs = pd.read_csv(
            path, sep=separator, header=None, usecols=[0, 1, 2],
            names=["id_a", "id_b", "distance"],
        )
    return pairs[["id_a", "id_b", "distance"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--hosts", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.00123693)
    parser.add_argument("--separator", default="\t")
    args = parser.parse_args()

    hosts = pd.read_csv(args.hosts, dtype=str)
    required = {"plasmid_accession", "host_genus"}
    if not required.issubset(hosts.columns):
        raise ValueError(f"Host table must contain {sorted(required)}")
    host_map = hosts.drop_duplicates("plasmid_accession").set_index("plasmid_accession")["host_genus"].to_dict()
    cohort = pd.read_csv(args.cohort, dtype=str)
    ids = cohort["plasmid_accession"].astype(str).tolist()
    cohort_set = set(ids)

    pairs = read_pairs(args.pairs, args.separator)
    pairs["distance"] = pd.to_numeric(pairs["distance"], errors="coerce")
    pairs = pairs[
        (pairs.distance <= args.cutoff)
        & pairs.id_a.ne(pairs.id_b)
        & (pairs.id_a.isin(cohort_set) | pairs.id_b.isin(cohort_set))
    ].copy()
    pairs["genus_a"] = pairs.id_a.map(host_map)
    pairs["genus_b"] = pairs.id_b.map(host_map)
    genus_a = pairs.genus_a.fillna("").str.strip()
    genus_b = pairs.genus_b.fillna("").str.strip()
    pairs = pairs[
        ~genus_a.str.lower().isin(UNRESOLVED)
        & ~genus_b.str.lower().isin(UNRESOLVED)
        & genus_a.ne(genus_b)
    ].copy()

    rows = []
    for accession in ids:
        left = pairs[pairs.id_a.eq(accession)].rename(
            columns={"id_b": "partner_accession", "genus_b": "partner_genus"}
        )
        right = pairs[pairs.id_b.eq(accession)].rename(
            columns={"id_a": "partner_accession", "genus_a": "partner_genus"}
        )
        evidence = pd.concat(
            [left[["partner_accession", "partner_genus", "distance"]],
             right[["partner_accession", "partner_genus", "distance"]]],
            ignore_index=True,
        ).sort_values(["distance", "partner_accession"])
        rows.append({
            "plasmid_accession": accession,
            "host_genus": host_map.get(accession, ""),
            "cross_genus_transfer": int(not evidence.empty),
            "qualifying_partner_count": int(evidence.partner_accession.nunique()) if not evidence.empty else 0,
            "nearest_qualifying_partner": evidence.iloc[0].partner_accession if not evidence.empty else "",
            "nearest_qualifying_partner_genus": evidence.iloc[0].partner_genus if not evidence.empty else "",
            "nearest_qualifying_pair_distance": evidence.iloc[0].distance if not evidence.empty else "",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
