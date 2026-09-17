#!/usr/bin/env python3
"""Extract the Figure 3 source-data table from the full-test ladder results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
SOURCE_JSON = (
    REPO_ROOT
    / "crystal_language"
    / "eval_results"
    / "ladder_tierA_s2_full_neutron"
    / "ladder_test_table.json"
)
OUTPUT_CSV = FIGURE_DIR / "data" / "figure_3_source_data.csv"

EXPECTED_TEST_SIZE = 33_944
EXPECTED_CANDIDATES = 100
TOP_K = (1, 5, 10, 20, 50)

LATTICES = (
    ("lattice_free", "Lattice-free", 0),
    ("known_lattice", "Known lattice", 1),
)
CONDITIONS = (
    ("pxrd", "PXRD", 0),
    ("pxrd_neutron", "PXRD + ND", 1),
    ("pxrd_xrf", "PXRD + XRF", 2),
    ("pxrd_neutron_xrf", "PXRD + ND + XRF", 3),
    ("pxrd_formula", "Formula reference", 4),
)

FIELDNAMES = (
    "lattice_key",
    "lattice_label",
    "use_lattice",
    "condition_key",
    "condition_label",
    "condition_order",
    "n_records",
    "n_candidates",
    "metric",
    "rank",
    "composition_rate",
    "composition_rate_percent",
    "success_rate",
    "success_rate_percent",
)


def _rate(value: Any, field: str) -> float:
    """Return a validated rate in [0, 1]."""

    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], got {rate}")
    return rate


def extract_rows(payload: dict[str, Any]) -> list[dict[str, str | int]]:
    """Select the five manuscript conditions and reranked Top-k metrics."""

    table = payload.get("table")
    if not isinstance(table, dict):
        raise ValueError("Source JSON is missing the top-level 'table' object")

    rows: list[dict[str, str | int]] = []
    observed_n: set[int] = set()
    observed_candidates: set[int] = set()

    for lattice_key, lattice_label, use_lattice in LATTICES:
        lattice_table = table.get(lattice_key)
        if not isinstance(lattice_table, dict):
            raise ValueError(f"Missing lattice block: {lattice_key}")

        for condition_key, condition_label, condition_order in CONDITIONS:
            entry = lattice_table.get(condition_key)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Missing condition {condition_key!r} in {lattice_key!r}"
                )
            if entry.get("condition_set") != condition_key:
                raise ValueError(
                    f"condition_set mismatch for {lattice_key}/{condition_key}"
                )
            if int(entry.get("use_lattice", -1)) != use_lattice:
                raise ValueError(
                    f"use_lattice mismatch for {lattice_key}/{condition_key}"
                )

            n_records = int(entry["n_records"])
            observed_n.add(n_records)
            composition_rate = _rate(
                entry["comp_rate"], f"{lattice_key}/{condition_key}/comp_rate"
            )

            rerank = entry.get("topk_rerank")
            if not isinstance(rerank, dict) or not rerank:
                raise ValueError(
                    f"Missing topk_rerank for {lattice_key}/{condition_key}"
                )
            available_ranks = {int(rank) for rank in rerank}
            n_candidates = max(available_ranks)
            observed_candidates.add(n_candidates)

            missing_ranks = set(TOP_K) - available_ranks
            if missing_ranks:
                raise ValueError(
                    f"Missing rerank values {sorted(missing_ranks)} for "
                    f"{lattice_key}/{condition_key}"
                )

            for rank in TOP_K:
                success_rate = _rate(
                    rerank[str(rank)],
                    f"{lattice_key}/{condition_key}/topk_rerank/{rank}",
                )
                rows.append(
                    {
                        "lattice_key": lattice_key,
                        "lattice_label": lattice_label,
                        "use_lattice": use_lattice,
                        "condition_key": condition_key,
                        "condition_label": condition_label,
                        "condition_order": condition_order,
                        "n_records": n_records,
                        "n_candidates": n_candidates,
                        "metric": "topk_rerank",
                        "rank": rank,
                        "composition_rate": format(composition_rate, ".15g"),
                        "composition_rate_percent": f"{100 * composition_rate:.6f}",
                        "success_rate": format(success_rate, ".15g"),
                        "success_rate_percent": f"{100 * success_rate:.6f}",
                    }
                )

    if observed_n != {EXPECTED_TEST_SIZE}:
        raise ValueError(
            f"Expected full test N={EXPECTED_TEST_SIZE:,}, observed {observed_n}"
        )
    if observed_candidates != {EXPECTED_CANDIDATES}:
        raise ValueError(
            "Expected N=100 candidates/sample from the available rerank ranks, "
            f"observed {observed_candidates}"
        )

    expected_rows = len(LATTICES) * len(CONDITIONS) * len(TOP_K)
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, created {len(rows)}")
    return rows


def main() -> None:
    with SOURCE_JSON.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = extract_rows(payload)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows to {OUTPUT_CSV.relative_to(REPO_ROOT)} "
        f"(full test N={EXPECTED_TEST_SIZE:,}; "
        f"N={EXPECTED_CANDIDATES} candidates/sample; rerank only)."
    )


if __name__ == "__main__":
    main()
