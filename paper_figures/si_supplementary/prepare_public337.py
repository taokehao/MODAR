#!/usr/bin/env python3
"""Curate SI tables for the public-337 spectroscopic ladder (no formula MACE)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


SI_DIR = Path(__file__).resolve().parent
REPO_ROOT = SI_DIR.parents[1]
DATA_DIR = SI_DIR / "data"
SOURCE = REPO_ROOT / "crystal_language/eval_results/chgnet/ladder_public_si_report_spec_cm.json"

CONDS = (
    ("pxrd", "PXRD"),
    ("pxrd_neutron", "PXRD + ND"),
    ("pxrd_xrf", "PXRD + XRF"),
    ("pxrd_neutron_xrf", "PXRD + ND + XRF"),
    ("pxrd_neutron_xrf_xps", "PXRD + ND + XRF + XPS"),
)
STEPS = (
    ("step_pxrd_to_pxrd_xrf", "PXRD → +XRF"),
    ("step_pxrd_xrf_to_pxrd_neutron_xrf", "+XRF → +ND+XRF"),
    ("step_pxrd_to_pxrd_neutron", "PXRD → +ND"),
)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    blob = json.loads(SOURCE.read_text(encoding="utf-8"))
    ladder_rows: list[dict[str, Any]] = []
    for lat, lattice_label in ((1, "Known lattice"), (0, "Lattice-free")):
        for key, label in CONDS:
            row = blob[f"{key}_lat{lat}"]
            if int(row["n"]) != 337:
                raise ValueError(f"{key}_lat{lat} n={row['n']}")
            has_ref = "ref1" in row
            ladder_rows.append(
                {
                    "lattice": lattice_label,
                    "condition_key": key,
                    "condition_label": label,
                    "n": 337,
                    "top1": int(row["ref1"]) if has_ref else "",
                    "top1_ci_low": int(row["ref1_ci"][0]) if has_ref else "",
                    "top1_ci_high": int(row["ref1_ci"][1]) if has_ref else "",
                    "top20": int(row["ref20"]) if has_ref else "",
                    "top20_ci_low": int(row["ref20_ci"][0]) if has_ref else "",
                    "top20_ci_high": int(row["ref20_ci"][1]) if has_ref else "",
                    "oracle": int(row.get("ref_oracle", row["oracle"])),
                    "fus1": int(row["fus1"]),
                    "fus20": int(row["fus20"]),
                    "small_n": int(row["small_n"]) if has_ref else "",
                    "small_top1": int(row["small_ref1"]) if has_ref else "",
                    "small_top20": int(row["small_ref20"]) if has_ref else "",
                }
            )
    write_csv(
        DATA_DIR / "fig_s17_public337_ladder.csv",
        [
            "lattice",
            "condition_key",
            "condition_label",
            "n",
            "top1",
            "top1_ci_low",
            "top1_ci_high",
            "top20",
            "top20_ci_low",
            "top20_ci_high",
            "oracle",
            "fus1",
            "fus20",
            "small_n",
            "small_top1",
            "small_top20",
        ],
        ladder_rows,
    )
    size_rows = [row for row in ladder_rows if row["lattice"] == "Known lattice" and row["small_n"] != ""]
    write_csv(
        DATA_DIR / "fig_s18_public337_size_split.csv",
        [
            "condition_key",
            "condition_label",
            "n_all",
            "top1_all",
            "top20_all",
            "n_le20",
            "top1_le20",
            "top20_le20",
        ],
        [
            {
                "condition_key": row["condition_key"],
                "condition_label": row["condition_label"],
                "n_all": 337,
                "top1_all": row["top1"],
                "top20_all": row["top20"],
                "n_le20": row["small_n"],
                "top1_le20": row["small_top1"],
                "top20_le20": row["small_top20"],
            }
            for row in size_rows
            if row["condition_key"] != "pxrd_neutron_xrf_xps"
        ],
    )
    step_rows: list[dict[str, Any]] = []
    for lat, lattice_label in ((1, "Known lattice"), (0, "Lattice-free")):
        for key, label in STEPS:
            rec = blob[f"{key}_lat{lat}"]
            for metric in ("top1", "top20"):
                delta, lo, hi = rec[metric]
                step_rows.append(
                    {
                        "step": label,
                        "lattice": lattice_label,
                        "metric": "Top-1" if metric == "top1" else "Top-20",
                        "delta": int(delta),
                        "ci_low": int(lo),
                        "ci_high": int(hi),
                    }
                )
    write_csv(
        DATA_DIR / "fig_s19_public337_paired_steps.csv",
        ["step", "lattice", "metric", "delta", "ci_low", "ci_high"],
        step_rows,
    )
    xps_rows = [
        row
        for row in ladder_rows
        if row["condition_key"] in {"pxrd_neutron_xrf", "pxrd_neutron_xrf_xps"}
    ]
    write_csv(
        DATA_DIR / "fig_s20_public337_xps.csv",
        [
            "lattice",
            "condition_key",
            "condition_label",
            "n",
            "top1",
            "top1_ci_low",
            "top1_ci_high",
            "top20",
            "top20_ci_low",
            "top20_ci_high",
        ],
        [
            {
                "lattice": row["lattice"],
                "condition_key": row["condition_key"],
                "condition_label": "+ND+XRF" if row["condition_key"] == "pxrd_neutron_xrf" else "+ND+XRF+XPS",
                "n": row["n"],
                "top1": row["top1"],
                "top1_ci_low": row["top1_ci_low"],
                "top1_ci_high": row["top1_ci_high"],
                "top20": row["top20"],
                "top20_ci_low": row["top20_ci_low"],
                "top20_ci_high": row["top20_ci_high"],
            }
            for row in xps_rows
        ],
    )
    print("wrote fig_s17/s18/s19/s20 csv")


if __name__ == "__main__":
    main()
