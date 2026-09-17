#!/usr/bin/env python3
"""Prepare the curated source-data tables for manuscript Figure 5."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
DATA_DIR = FIGURE_DIR / "data"

SIZE_BIN_SOURCE = (
    REPO_ROOT
    / "crystal_language/eval_results/ladder_tierA_s2_full_neutron/size_bins_20.json"
)
BIGCELL_SOURCE = (
    REPO_ROOT
    / "crystal_language/eval_results/bigcell_ab/oracle_same_subset_comparison.json"
)
PLAN_SOURCE = REPO_ROOT / "总体实施方案.md"

SIZE_BIN_SOURCE_REL = SIZE_BIN_SOURCE.relative_to(REPO_ROOT).as_posix()
BIGCELL_SOURCE_REL = BIGCELL_SOURCE.relative_to(REPO_ROOT).as_posix()

TOP_KS = (1, 5, 10, 20, 50)
LATTICE_LABELS = {
    "lattice_free": "Lattice-free",
    "known_lattice": "Known lattice",
}
BIGCELL_LATTICE_LABELS = {
    "free": "Lattice-free",
    "known": "Known lattice",
}
METHODS = {
    "pxrd_neutron_xrf": "Base multimodal",
    "pxrd_formula": "Base formula reference",
}
BIGCELL_METHODS = {
    "base_pxrd_neutron_xrf": ("Base multimodal", "PXRD + ND + XRF"),
    "bigcell_pxrd_neutron_xrf": (
        "Large-cell specialist",
        "PXRD + ND + XRF",
    ),
    "base_pxrd_formula_oracle": ("Base formula reference", "PXRD + formula"),
}

EXPECTED_FULL_BIN_COUNTS = {"<=20": 30_159, ">20": 3_785}
EXPECTED_FULL_TEST = 33_944
EXPECTED_BIGCELL_SUBSET = 500
EXPECTED_TOTAL_TRAIN = 3_456_202
EXPECTED_BIGCELL_TRAIN = 362_673


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _extract_int(text: str, pattern: str, description: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not locate {description} in {PLAN_SOURCE}")
    return int(match.group(1).replace(",", ""))


def prepare_full_size_bins() -> list[dict[str, object]]:
    source = _read_json(SIZE_BIN_SOURCE)
    if source.get("split") != "test" or source.get("threshold") != 20:
        raise ValueError("Size-bin artifact is not the expected test split at 20 sites")

    rows: list[dict[str, object]] = []
    observed_counts: dict[str, set[int]] = {key: set() for key in EXPECTED_FULL_BIN_COUNTS}
    table = source["table"]
    for lattice_key, lattice_label in LATTICE_LABELS.items():
        for condition_key, method_label in METHODS.items():
            bins = table[lattice_key][condition_key]["bins"]
            for size_bin in ("<=20", ">20"):
                record = bins[size_bin]
                n_records = int(record["n_records"])
                observed_counts[size_bin].add(n_records)
                for top_k in TOP_KS:
                    rows.append(
                        {
                            "cohort": "Full test size bin",
                            "lattice": lattice_label,
                            "size_bin": size_bin,
                            "n": n_records,
                            "method": method_label,
                            "top_k": top_k,
                            "match_rate_percent": (
                                f"{100.0 * float(record['topk_rerank'][str(top_k)]):.6f}"
                            ),
                            "source_artifact": SIZE_BIN_SOURCE_REL,
                        }
                    )

    for size_bin, expected in EXPECTED_FULL_BIN_COUNTS.items():
        if observed_counts[size_bin] != {expected}:
            raise ValueError(
                f"Inconsistent {size_bin} count: {observed_counts[size_bin]} "
                f"(expected {expected})"
            )
    if sum(EXPECTED_FULL_BIN_COUNTS.values()) != EXPECTED_FULL_TEST:
        raise ValueError("Full-test bin counts do not sum to the expected total")
    return rows


def prepare_training_capacity() -> list[dict[str, object]]:
    plan_text = PLAN_SOURCE.read_text(encoding="utf-8")
    section_match = re.search(
        r"### 8\.10\.1\b(?P<section>.*?)(?=\n### 8\.10\.2\b)",
        plan_text,
        flags=re.DOTALL,
    )
    if section_match is None:
        raise ValueError("Could not isolate 总体实施方案.md §8.10.1")

    total_train = _extract_int(
        plan_text,
        r"tight 版 Tier-A 去重后构成（train\s+([\d,]+)\s*/\s*valid",
        "Tier-A total train count",
    )
    bigcell_train = _extract_int(
        section_match.group("section"),
        r"Tier-A train 中 `>20` 大胞 \*\*([\d,]+)\*\* 条",
        "large-cell train count in §8.10.1",
    )
    if total_train != EXPECTED_TOTAL_TRAIN:
        raise ValueError(
            f"Unexpected total train count {total_train}; expected {EXPECTED_TOTAL_TRAIN}"
        )
    if bigcell_train != EXPECTED_BIGCELL_TRAIN:
        raise ValueError(
            f"Unexpected >20 train count {bigcell_train}; "
            f"expected {EXPECTED_BIGCELL_TRAIN}"
        )

    smallcell_train = total_train - bigcell_train
    records = [
        ("<=20", smallcell_train, "Base model"),
        (">20", bigcell_train, "Large-cell specialist"),
    ]
    return [
        {
            "size_bin": size_bin,
            "n_train": count,
            "share_percent": f"{100.0 * count / total_train:.6f}",
            "routed_model": routed_model,
            "provenance": (
                "总体实施方案.md §8.10.1 (>20 count and routing); "
                "§8.7.1 (total train count)"
            ),
        }
        for size_bin, count, routed_model in records
    ]


def prepare_same_subset() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    source = _read_json(BIGCELL_SOURCE)
    if int(source.get("n", -1)) != EXPECTED_BIGCELL_SUBSET:
        raise ValueError("Big-cell comparison is not the expected N=500 subset")
    if tuple(int(value) for value in source.get("top_ks", [])) != TOP_KS:
        raise ValueError("Big-cell comparison has unexpected Top-k coordinates")

    result_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    for lattice_key, lattice_label in BIGCELL_LATTICE_LABELS.items():
        lattice_results = source[lattice_key]
        for source_key, (method_label, input_label) in BIGCELL_METHODS.items():
            values = lattice_results[source_key]
            if len(values) != len(TOP_KS):
                raise ValueError(f"Unexpected result length for {lattice_key}/{source_key}")
            for top_k, value in zip(TOP_KS, values, strict=True):
                result_rows.append(
                    {
                        "cohort": "Same 500 large-cell subset",
                        "lattice": lattice_label,
                        "n": EXPECTED_BIGCELL_SUBSET,
                        "method": method_label,
                        "input": input_label,
                        "top_k": top_k,
                        "match_rate_percent": f"{float(value):.6f}",
                        "source_artifact": BIGCELL_SOURCE_REL,
                    }
                )

        specialist = lattice_results["bigcell_pxrd_neutron_xrf"]
        comparators = (
            (
                "Specialist - base multimodal",
                lattice_results["base_pxrd_neutron_xrf"],
            ),
            (
                "Specialist - base formula reference",
                lattice_results["base_pxrd_formula_oracle"],
            ),
        )
        for comparison, comparator in comparators:
            for top_k, specialist_value, comparator_value in zip(
                TOP_KS,
                specialist,
                comparator,
                strict=True,
            ):
                gain_rows.append(
                    {
                        "cohort": "Same 500 large-cell subset",
                        "lattice": lattice_label,
                        "n": EXPECTED_BIGCELL_SUBSET,
                        "comparison": comparison,
                        "top_k": top_k,
                        "gain_pp": f"{float(specialist_value) - float(comparator_value):.6f}",
                        "source_artifact": BIGCELL_SOURCE_REL,
                    }
                )
    return result_rows, gain_rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_rows = prepare_full_size_bins()
    capacity_rows = prepare_training_capacity()
    same_subset_rows, gain_rows = prepare_same_subset()

    _write_csv(
        DATA_DIR / "full_test_size_bins.csv",
        [
            "cohort",
            "lattice",
            "size_bin",
            "n",
            "method",
            "top_k",
            "match_rate_percent",
            "source_artifact",
        ],
        full_rows,
    )
    _write_csv(
        DATA_DIR / "train_capacity.csv",
        [
            "size_bin",
            "n_train",
            "share_percent",
            "routed_model",
            "provenance",
        ],
        capacity_rows,
    )
    _write_csv(
        DATA_DIR / "bigcell_same_subset.csv",
        [
            "cohort",
            "lattice",
            "n",
            "method",
            "input",
            "top_k",
            "match_rate_percent",
            "source_artifact",
        ],
        same_subset_rows,
    )
    _write_csv(
        DATA_DIR / "bigcell_gains.csv",
        [
            "cohort",
            "lattice",
            "n",
            "comparison",
            "top_k",
            "gain_pp",
            "source_artifact",
        ],
        gain_rows,
    )
    print(
        "Prepared Figure 5 data: "
        f"{len(full_rows)} full-bin rows, {len(capacity_rows)} capacity rows, "
        f"{len(same_subset_rows)} same-subset rows, and {len(gain_rows)} gain rows."
    )


if __name__ == "__main__":
    main()
