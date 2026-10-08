"""Apply the frozen annotation-to-feature rules used after harmonization.

This utility does not run AMRFinderPlus, MOB-suite, or CONJScan. It converts
their frozen tabular outputs (or CDS product text) into model features.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_rules(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def present(value) -> int:
    return int(value is not None and str(value).strip().lower() not in {"", "-", "nan", "none"})


class Matcher:
    def __init__(self, rules):
        self.is_patterns = {
            key: re.compile(value, re.IGNORECASE)
            for key, value in rules["IS_count"]["patterns"].items()
        }
        self.esbl = re.compile(rules["AMRFinderPlus"]["has_ESBL"]["element_name_regex"], re.IGNORECASE)
        self.carb = re.compile(rules["AMRFinderPlus"]["has_carbapenemase"]["subclass_token_regex"])

    def is_hits(self, product):
        value = "" if product is None else str(product)
        return [name for name, pattern in self.is_patterns.items() if pattern.search(value)]

    def amr_flags(self, row):
        is_amr = str(row.get("Type", "")).strip() == "AMR"
        subclass = str(row.get("Subclass", "")).strip().upper()
        name = str(row.get("Element name", "")).strip()
        return {
            "retained_as_ARG": int(is_amr),
            "classified_carbapenemase": int(is_amr and bool(self.carb.search(subclass))),
            "classified_ESBL": int(is_amr and bool(self.esbl.search(name))),
        }

    @staticmethod
    def mob_features(row):
        return {
            "has_oriT": present(row.get("orit_type(s)")),
            "has_relaxase": present(row.get("relaxase_type(s)")),
            "has_MPF": present(row.get("mpf_type")),
            "mobility": int(str(row.get("predicted_mobility", "")).strip().lower() == "conjugative"),
        }


def read_rows(path, delimiter):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cds", "amrfinder", "mob"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delimiter", choices=["comma", "tab"], default="comma")
    parser.add_argument("--rules", type=Path, default=HERE / "rules.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; this utility never overwrites results.")

    matcher = Matcher(load_rules(args.rules))
    rows = list(read_rows(args.input, "," if args.delimiter == "comma" else "\t"))
    details, summary = [], {}
    for index, row in enumerate(rows, 2):
        accession = str(row.get("plasmid_accession") or row.get("sample_id") or "").strip()
        if not accession:
            parser.error(f"Missing plasmid_accession/sample_id at row {index}")
        item = {"input_row": index, "plasmid_accession": accession}
        agg = summary.setdefault(accession, {})
        if args.mode == "cds":
            hits = matcher.is_hits(row.get("product"))
            item.update(IS_match=int(bool(hits)), IS_rule_hits=hits)
            agg["IS_count"] = agg.get("IS_count", 0) + int(bool(hits))
        elif args.mode == "amrfinder":
            flags = matcher.amr_flags(row)
            item.update(flags)
            try:
                length = int(float(row.get("plasmid_length", "")))
            except (TypeError, ValueError):
                parser.error(f"AMRFinder mode requires a positive plasmid_length at row {index}")
            if length <= 0:
                parser.error(f"AMRFinder mode requires a positive plasmid_length at row {index}")
            agg.setdefault("_lengths", set()).add(length)
            if flags["retained_as_ARG"]:
                agg["total_arg_count"] = agg.get("total_arg_count", 0) + 1
                value = str(row.get("Class", "")).strip()
                if value:
                    agg.setdefault("_classes", set()).add(value)
            for key in ("classified_carbapenemase", "classified_ESBL"):
                agg[key] = max(agg.get(key, 0), flags[key])
        else:
            flags = matcher.mob_features(row)
            if agg:
                parser.error(f"MOB input must contain one row per accession: duplicate {accession}")
            item.update(flags)
            agg.update(flags)
        details.append(item)
    for agg in summary.values():
        if args.mode == "amrfinder":
            lengths = agg.pop("_lengths", set())
            if len(lengths) != 1:
                parser.error("Each accession must have one consistent plasmid_length")
            agg["ARG_class_count"] = len(agg.pop("_classes", set()))
            agg.setdefault("total_arg_count", 0)
            agg["ARG_density"] = round(agg["total_arg_count"] / (next(iter(lengths)) / 1000), 4)
            agg["has_carbapenemase"] = agg.pop("classified_carbapenemase", 0)
            agg["has_ESBL"] = agg.pop("classified_ESBL", 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"mode": args.mode, "per_plasmid": summary, "per_annotation": details}, handle,
                  ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
