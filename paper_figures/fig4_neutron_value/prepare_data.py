#!/usr/bin/env python3
"""Prepare the source-data tables for manuscript Figure 4."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
SOURCE_JSON = (
    REPO_ROOT
    / "crystal_language/eval_results/ladder_tierA_s2_10k/stats/ladder_stats.json"
)
DATA_DIR = FIGURE_DIR / "data"

SYSTEM_ORDER = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]
LATTICE_MODES = [
    ("lattice_free", "Lattice-free"),
    ("known_lattice", "Known lattice"),
]
PAIRED_COMPARISONS = [
    ("neutron_vs_pxrd", "ND vs PXRD"),
    ("neutron_given_xrf", "ND given XRF"),
]


def _percentage_points(value: float) -> float:
    return round(100.0 * float(value), 4)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare() -> tuple[Path, Path]:
    with SOURCE_JSON.open(encoding="utf-8") as handle:
        stats = json.load(handle)

    n_boot = int(stats["n_boot"])
    if n_boot <= 0:
        raise ValueError("Expected a positive number of bootstrap resamples")

    paired_rows: list[dict[str, Any]] = []
    for comparison_order, (comparison_key, comparison_label) in enumerate(
        PAIRED_COMPARISONS
    ):
        for lattice_order, (lattice_key, lattice_label) in enumerate(LATTICE_MODES):
            effect = stats["paired_diffs"][lattice_key][comparison_key]
            if int(effect["n"]) != 10_000:
                raise ValueError(
                    f"{lattice_key}/{comparison_key} has N={effect['n']}, expected 10000"
                )
            low_pp = _percentage_points(effect["lo"])
            high_pp = _percentage_points(effect["hi"])
            if low_pp > 0:
                interpretation = "significant positive"
            elif high_pp < 0:
                interpretation = "significant negative"
            else:
                interpretation = "CI crosses zero"
            paired_rows.append(
                {
                    "comparison_order": comparison_order,
                    "lattice_order": lattice_order,
                    "comparison": comparison_label,
                    "lattice_mode": lattice_label,
                    "estimate_pp": _percentage_points(effect["diff"]),
                    "ci_low_pp": low_pp,
                    "ci_high_pp": high_pp,
                    "n": int(effect["n"]),
                    "n_boot": n_boot,
                    "ci_level": 0.95,
                    "ci_scope": "aligned 10k evaluation subset",
                    "interpretation": interpretation,
                }
            )

    crystal_system_records = stats["groups"]["crystal_system"]
    records_by_key = {
        (record["lattice"], record["group"]): record
        for record in crystal_system_records
    }
    expected_keys = {
        (lattice_key, system)
        for lattice_key, _ in LATTICE_MODES
        for system in SYSTEM_ORDER
    }
    missing = expected_keys.difference(records_by_key)
    if missing:
        raise ValueError(f"Missing crystal-system records: {sorted(missing)}")

    system_rows: list[dict[str, Any]] = []
    for lattice_order, (lattice_key, lattice_label) in enumerate(LATTICE_MODES):
        for system_order, system in enumerate(SYSTEM_ORDER):
            record = records_by_key[(lattice_key, system)]
            system_rows.append(
                {
                    "system_order": system_order,
                    "lattice_order": lattice_order,
                    "crystal_system": system,
                    "lattice_mode": lattice_label,
                    "n": int(record["n"]),
                    "pxrd_top1_pct": _percentage_points(record["pxrd"]),
                    "neutron_top1_pct": _percentage_points(record["pxrd_neutron"]),
                    "neutron_gain_pp": _percentage_points(record["d_neutron"]),
                    "xrf_top1_pct": _percentage_points(record["pxrd_xrf"]),
                    "neutron_xrf_top1_pct": _percentage_points(
                        record["pxrd_neutron_xrf"]
                    ),
                    "neutron_given_xrf_gain_pp": _percentage_points(
                        record["d_neutron_xrf"]
                    ),
                }
            )

    paired_path = DATA_DIR / "paired_neutron_effects.csv"
    systems_path = DATA_DIR / "crystal_system_neutron_effects.csv"
    _write_csv(
        paired_path,
        paired_rows,
        [
            "comparison_order",
            "lattice_order",
            "comparison",
            "lattice_mode",
            "estimate_pp",
            "ci_low_pp",
            "ci_high_pp",
            "n",
            "n_boot",
            "ci_level",
            "ci_scope",
            "interpretation",
        ],
    )
    _write_csv(
        systems_path,
        system_rows,
        [
            "system_order",
            "lattice_order",
            "crystal_system",
            "lattice_mode",
            "n",
            "pxrd_top1_pct",
            "neutron_top1_pct",
            "neutron_gain_pp",
            "xrf_top1_pct",
            "neutron_xrf_top1_pct",
            "neutron_given_xrf_gain_pp",
        ],
    )
    return paired_path, systems_path


if __name__ == "__main__":
    outputs = prepare()
    for output in outputs:
        print(output.relative_to(REPO_ROOT))
