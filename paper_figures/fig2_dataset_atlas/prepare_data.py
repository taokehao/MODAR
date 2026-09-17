#!/usr/bin/env python3
"""Prepare the reproducible data products used by manuscript Figure 2.

The test-set statistics are computed by streaming every JSONL record.  No
sampling or cached metadata is used for element, size, or symmetry statistics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"

METADATA_SUMMARY = (
    REPO_ROOT
    / "data_prep"
    / "outputs"
    / "metadata_index_unified_tight"
    / "metadata_index_summary.json"
)
EXPORT_SUMMARY = (
    REPO_ROOT
    / "data_prep"
    / "outputs"
    / "crystal_language_unified_tierA_v2"
    / "export_summary.json"
)
TEST_JSONL = (
    REPO_ROOT
    / "data_prep"
    / "outputs"
    / "crystal_language_unified_tierA_v2"
    / "test.jsonl"
)

FORMULA_RE = re.compile(
    r"<FORMULA>\s+(.*?)\s+<N_SITES>\s+((?:\d\s*)+?)<SPACE_GROUP>"
)
SPACE_GROUP_RE = re.compile(r"<SG(\d{3})>")
ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
    "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
    "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
    "Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ELEMENT_SET = set(ELEMENTS)

CRYSTAL_SYSTEMS = (
    ("triclinic", 1, 2),
    ("monoclinic", 3, 15),
    ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142),
    ("trigonal", 143, 167),
    ("hexagonal", 168, 194),
    ("cubic", 195, 230),
)


def load_json_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_formula_and_target_n_sites(target: str, line_number: int) -> tuple[set[str], int]:
    match = FORMULA_RE.search(target)
    if match is None:
        raise ValueError(f"line {line_number}: missing <FORMULA> ... <N_SITES> segment")

    formula_tokens = match.group(1).split()
    if not formula_tokens:
        raise ValueError(f"line {line_number}: malformed formula token sequence")

    elements: set[str] = set()
    index = 0
    while index < len(formula_tokens):
        symbol = formula_tokens[index]
        if not ELEMENT_RE.fullmatch(symbol) or symbol not in ELEMENT_SET:
            raise ValueError(f"line {line_number}: invalid element symbol {symbol!r}")
        amount_start = index + 1
        index = amount_start
        while index < len(formula_tokens) and formula_tokens[index] not in ELEMENT_SET:
            index += 1
        if index == amount_start:
            raise ValueError(f"line {line_number}: formula amount missing for {symbol}")
        # Formula integers are also tokenized digit by digit (16 -> "1 6").
        amount_text = "".join(formula_tokens[amount_start:index])
        try:
            amount = Decimal(amount_text)
        except InvalidOperation as exc:
            raise ValueError(
                f"line {line_number}: invalid formula amount {amount_text!r}"
            ) from exc
        if amount <= 0:
            raise ValueError(f"line {line_number}: non-positive formula amount")
        elements.add(symbol)

    # Integer targets are tokenized digit by digit (for example, 12 -> "1 2").
    return elements, int("".join(match.group(2).split()))


def crystal_system(space_group: int) -> str:
    for name, lower, upper in CRYSTAL_SYSTEMS:
        if lower <= space_group <= upper:
            return name
    raise ValueError(f"space-group number outside the standard 1-230 range: {space_group}")


def percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def main() -> None:
    for required_path in (METADATA_SUMMARY, EXPORT_SUMMARY, TEST_JSONL):
        if not required_path.is_file():
            raise FileNotFoundError(f"required source data not found: {required_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata, metadata_sha256 = load_json_with_hash(METADATA_SUMMARY)
    export, export_sha256 = load_json_with_hash(EXPORT_SUMMARY)

    counts = metadata["counts"]
    splits = {
        split: int(export["splits"][split]["written"])
        for split in ("train", "valid", "test")
    }
    raw_count = int(counts["records"])
    tier_a_count = int(counts["tier_a"])
    deduplicated_count = int(counts["tier_a_kept_after_dedup"])
    split_total = sum(splits.values())
    if split_total != deduplicated_count:
        raise ValueError(
            "split total does not equal metadata Tier-A kept-after-dedup count: "
            f"{split_total:,} != {deduplicated_count:,}"
        )

    source_counts = {
        str(source): int(count)
        for source, count in metadata["tier_a_kept_by_source"].items()
    }
    if sum(source_counts.values()) != deduplicated_count:
        raise ValueError("Tier-A kept source counts do not sum to the deduplicated total")

    funnel_rows = [
        {
            "step": 1,
            "stage": "Unified multimodal records",
            "count": raw_count,
            "display_count": compact_number(raw_count),
            "retained_vs_raw_pct": "100.000",
            "retained_vs_previous_pct": "100.000",
        },
        {
            "step": 2,
            "stage": "Tier-A quality filter",
            "count": tier_a_count,
            "display_count": compact_number(tier_a_count),
            "retained_vs_raw_pct": f"{100 * tier_a_count / raw_count:.3f}",
            "retained_vs_previous_pct": f"{100 * tier_a_count / raw_count:.3f}",
        },
        {
            "step": 3,
            "stage": "After tight deduplication",
            "count": deduplicated_count,
            "display_count": compact_number(deduplicated_count),
            "retained_vs_raw_pct": f"{100 * deduplicated_count / raw_count:.3f}",
            "retained_vs_previous_pct": f"{100 * deduplicated_count / tier_a_count:.3f}",
        },
        {
            "step": 4,
            "stage": "Frozen train/valid/test split",
            "count": split_total,
            "display_count": compact_number(split_total),
            "retained_vs_raw_pct": f"{100 * split_total / raw_count:.3f}",
            "retained_vs_previous_pct": "100.000",
        },
    ]
    write_csv(
        DATA_DIR / "data_governance_funnel.csv",
        [
            "step",
            "stage",
            "count",
            "display_count",
            "retained_vs_raw_pct",
            "retained_vs_previous_pct",
        ],
        funnel_rows,
    )

    source_rows = [
        {
            "source": source,
            "count": count,
            "fraction_pct": f"{100 * count / deduplicated_count:.6f}",
        }
        for source, count in sorted(
            source_counts.items(), key=lambda item: item[1], reverse=True
        )
    ]
    write_csv(
        DATA_DIR / "source_composition.csv",
        ["source", "count", "fraction_pct"],
        source_rows,
    )

    split_rows = [
        {
            "split": split,
            "count": count,
            "fraction_pct": f"{100 * count / split_total:.6f}",
        }
        for split, count in splits.items()
    ]
    write_csv(
        DATA_DIR / "split_sizes.csv",
        ["split", "count", "fraction_pct"],
        split_rows,
    )

    expected_test_count = splits["test"]
    element_record_counts: Counter[str] = Counter()
    n_sites_counts: Counter[int] = Counter()
    n_elements_counts: Counter[int] = Counter()
    system_counts: Counter[str] = Counter()
    n_sites_values: list[int] = []
    n_elements_values: list[int] = []
    test_sha256 = hashlib.sha256()
    actual_test_count = 0

    with TEST_JSONL.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            test_sha256.update(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            actual_test_count += 1

            if record.get("split") != "test":
                raise ValueError(
                    f"line {line_number}: expected split='test', got {record.get('split')!r}"
                )

            target = record.get("target")
            if not isinstance(target, str):
                raise ValueError(f"line {line_number}: target is not a string")
            elements, target_n_sites = parse_formula_and_target_n_sites(
                target, line_number
            )

            try:
                field_n_sites = int(record["n_sites"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: invalid n_sites field") from exc
            if field_n_sites <= 0:
                raise ValueError(f"line {line_number}: n_sites must be positive")
            if target_n_sites != field_n_sites:
                raise ValueError(
                    f"line {line_number}: target <N_SITES>={target_n_sites} "
                    f"does not match n_sites field={field_n_sites}"
                )

            space_group_matches = SPACE_GROUP_RE.findall(target)
            if len(space_group_matches) != 1:
                raise ValueError(
                    f"line {line_number}: expected exactly one <SG###> token"
                )
            space_group = int(space_group_matches[0])
            system = crystal_system(space_group)

            element_record_counts.update(elements)
            n_elements = len(elements)
            n_sites_counts[field_n_sites] += 1
            n_elements_counts[n_elements] += 1
            system_counts[system] += 1
            n_sites_values.append(field_n_sites)
            n_elements_values.append(n_elements)

            if actual_test_count % 5_000 == 0:
                print(f"Scanned {actual_test_count:,} test records")

    if actual_test_count != expected_test_count:
        raise ValueError(
            f"test.jsonl contains {actual_test_count:,} records; "
            f"export summary reports {expected_test_count:,}"
        )
    if sum(system_counts.values()) != actual_test_count:
        raise AssertionError("crystal-system counts do not cover the full test set")

    element_rows = [
        {
            "atomic_number": atomic_number,
            "element": element,
            "record_count": element_record_counts[element],
            "prevalence_pct": (
                f"{100 * element_record_counts[element] / actual_test_count:.6f}"
            ),
        }
        for atomic_number, element in enumerate(ELEMENTS, start=1)
    ]
    write_csv(
        DATA_DIR / "element_prevalence.csv",
        ["atomic_number", "element", "record_count", "prevalence_pct"],
        element_rows,
    )

    write_csv(
        DATA_DIR / "n_sites_distribution.csv",
        ["n_sites", "record_count", "fraction_pct"],
        (
            {
                "n_sites": value,
                "record_count": n_sites_counts[value],
                "fraction_pct": (
                    f"{100 * n_sites_counts[value] / actual_test_count:.6f}"
                ),
            }
            for value in range(min(n_sites_counts), max(n_sites_counts) + 1)
        ),
    )
    write_csv(
        DATA_DIR / "n_elements_distribution.csv",
        ["n_elements", "record_count", "fraction_pct"],
        (
            {
                "n_elements": value,
                "record_count": n_elements_counts[value],
                "fraction_pct": (
                    f"{100 * n_elements_counts[value] / actual_test_count:.6f}"
                ),
            }
            for value in range(min(n_elements_counts), max(n_elements_counts) + 1)
        ),
    )

    system_rows = [
        {
            "crystal_system": name,
            "sg_range": f"{lower}-{upper}",
            "record_count": system_counts[name],
            "fraction_pct": f"{100 * system_counts[name] / actual_test_count:.6f}",
        }
        for name, lower, upper in CRYSTAL_SYSTEMS
    ]
    write_csv(
        DATA_DIR / "crystal_system_distribution.csv",
        ["crystal_system", "sg_range", "record_count", "fraction_pct"],
        system_rows,
    )

    sorted_n_sites = sorted(n_sites_values)
    sorted_n_elements = sorted(n_elements_values)
    test_statistics = {
        "record_count": actual_test_count,
        "distinct_elements_present": sum(
            count > 0 for count in element_record_counts.values()
        ),
        "n_sites": {
            "min": min(sorted_n_sites),
            "median": percentile(sorted_n_sites, 0.5),
            "mean": sum(sorted_n_sites) / actual_test_count,
            "p95": percentile(sorted_n_sites, 0.95),
            "max": max(sorted_n_sites),
        },
        "n_elements": {
            "min": min(sorted_n_elements),
            "median": percentile(sorted_n_elements, 0.5),
            "mean": sum(sorted_n_elements) / actual_test_count,
            "p95": percentile(sorted_n_elements, 0.95),
            "max": max(sorted_n_elements),
        },
    }
    dataset_summary = {
        "figure": "Figure 2 dataset atlas",
        "funnel": funnel_rows,
        "tier_a_kept_source_composition": source_rows,
        "split_sizes": split_rows,
        "test_statistics": test_statistics,
        "crystal_system_distribution": system_rows,
        "definitions": {
            "element_prevalence": (
                "percentage of test records whose target <FORMULA> segment "
                "contains the element; each element is counted at most once per record"
            ),
            "n_sites": (
                "integer n_sites record field, cross-checked against target <N_SITES>"
            ),
            "n_elements": (
                "number of distinct element symbols parsed from target <FORMULA>"
            ),
            "crystal_system": (
                "mapped from the target <SG###> token using standard space-group "
                "number ranges: 1-2, 3-15, 16-74, 75-142, 143-167, "
                "168-194, and 195-230"
            ),
        },
        "validation": {
            "expected_test_records": expected_test_count,
            "scanned_test_records": actual_test_count,
            "sampling": "none; complete streaming scan",
            "target_n_sites_cross_check": "all records matched",
            "space_group_coverage": "all records mapped to SG 1-230",
        },
    }
    write_json(DATA_DIR / "dataset_summary.json", dataset_summary)

    provenance = {
        "inputs": [
            {
                "role": "governance funnel and kept source composition",
                "path_relative_to_repo": str(METADATA_SUMMARY.relative_to(REPO_ROOT)),
                "sha256": metadata_sha256,
            },
            {
                "role": "frozen split sizes",
                "path_relative_to_repo": str(EXPORT_SUMMARY.relative_to(REPO_ROOT)),
                "sha256": export_sha256,
            },
            {
                "role": "complete test-set scan",
                "path_relative_to_repo": str(TEST_JSONL.relative_to(REPO_ROOT)),
                "sha256": test_sha256.hexdigest(),
                "records_scanned": actual_test_count,
            },
        ],
        "script": "paper_figures/fig2_dataset_atlas/prepare_data.py",
        "sampling": "none",
        "outputs": sorted(
            {path.name for path in DATA_DIR.iterdir() if path.is_file()}
            | {"provenance.json"}
        ),
    }
    write_json(DATA_DIR / "provenance.json", provenance)

    print(
        "Prepared Figure 2 data: "
        f"{actual_test_count:,} test records, "
        f"{test_statistics['distinct_elements_present']} elements present"
    )
    print(f"Wrote {len(list(DATA_DIR.iterdir()))} files to {DATA_DIR}")


if __name__ == "__main__":
    main()
