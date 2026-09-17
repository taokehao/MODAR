#!/usr/bin/env python3
"""Curate source-data tables for manuscript Figure 7 (experimental trilogy)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
DATA_DIR = FIGURE_DIR / "data"
PANEL_DIR = DATA_DIR / "panels"
CHGNET = REPO_ROOT / "crystal_language/eval_results/chgnet"

SOURCES = {
    "cnrs": CHGNET / "rps_cnrs126_bootstrap_refined.json",
    "rruff": CHGNET / "rps_rruff269_bootstrap_refined.json",
    "public337": CHGNET / "ladder_public_si_report_spec_cm.json",
    "licoo2_spectra": DATA_DIR / "licoo2_spectra.csv",
    "licoo2_ranks": FIGURE_DIR / "data/panels/fig7c_licoo2_ranks.csv",
}

SPECTROSCOPIC = ("pxrd", "pxrd_neutron", "pxrd_xrf", "pxrd_neutron_xrf")
SPECTROSCOPIC_LABELS = {
    "pxrd": "PXRD",
    "pxrd_neutron": "PXRD + ND",
    "pxrd_xrf": "PXRD + XRF",
    "pxrd_neutron_xrf": "PXRD + ND + XRF",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ci_int(value: Any) -> int:
    return int(round(float(value)))


def prepare_rps() -> Path:
    rows: list[dict[str, Any]] = []
    for set_name, expected_n in (("cnrs", 126), ("rruff", 269)):
        blob = load_json(SOURCES[set_name])
        if int(blob["n"]) != expected_n:
            raise ValueError(f"{set_name}: expected n={expected_n}, got {blob['n']}")
        for metric in ("Top-1", "Top-20"):
            cell = blob[metric]
            rows.append(
                {
                    "set": set_name.upper() if set_name == "cnrs" else "RRUFF",
                    "n": expected_n,
                    "metric": metric,
                    "ours": int(cell["ours"]),
                    "ours_ci_low": ci_int(cell["ours_ci"][0]),
                    "ours_ci_high": ci_int(cell["ours_ci"][1]),
                    "rps_repro": int(cell["rps_repro"]),
                    "rps_reported": int(cell["rps_reported"]),
                    "diff": int(cell["diff"]),
                    "diff_ci_low": ci_int(cell["diff_ci"][0]),
                    "diff_ci_high": ci_int(cell["diff_ci"][1]),
                    "mcnemar_p": f"{float(cell['mcnemar_p']):.4g}",
                }
            )
    path = DATA_DIR / "fig7a_rps_head_to_head.csv"
    write_csv(
        path,
        [
            "set",
            "n",
            "metric",
            "ours",
            "ours_ci_low",
            "ours_ci_high",
            "rps_repro",
            "rps_reported",
            "diff",
            "diff_ci_low",
            "diff_ci_high",
            "mcnemar_p",
        ],
        rows,
    )
    shutil.copy(path, PANEL_DIR / "fig7a_rps_head_to_head.csv")
    return path


def prepare_public337() -> Path:
    blob = load_json(SOURCES["public337"])
    rows: list[dict[str, Any]] = []
    for cond in SPECTROSCOPIC:
        key = f"{cond}_lat1"
        row = blob[key]
        if int(row["n"]) != 337:
            raise ValueError(f"{key}: expected n=337, got {row['n']}")
        rows.append(
            {
                "condition_key": cond,
                "condition_label": SPECTROSCOPIC_LABELS[cond],
                "n": 337,
                "protocol": "chgnet_union_mace",
                "top1": int(row["ref1"]),
                "top1_ci_low": ci_int(row["ref1_ci"][0]),
                "top1_ci_high": ci_int(row["ref1_ci"][1]),
                "top20": int(row["ref20"]),
                "top20_ci_low": ci_int(row["ref20_ci"][0]),
                "top20_ci_high": ci_int(row["ref20_ci"][1]),
                "oracle": int(row["ref_oracle"]),
                "small_n": int(row["small_n"]),
                "small_top1": int(row["small_ref1"]),
                "small_top20": int(row["small_ref20"]),
                "small_oracle": int(row["small_oracle"]),
            }
        )
    formula = blob["pxrd_formula_lat1"]
    rows.append(
        {
            "condition_key": "pxrd_formula",
            "condition_label": "Formula reference",
            "n": 337,
            "protocol": "unrefined_fusion",
            "top1": int(formula["fus1"]),
            "top1_ci_low": "",
            "top1_ci_high": "",
            "top20": int(formula["fus20"]),
            "top20_ci_low": "",
            "top20_ci_high": "",
            "oracle": int(formula["oracle"]),
            "small_n": "",
            "small_top1": "",
            "small_top20": "",
            "small_oracle": "",
        }
    )
    path = DATA_DIR / "fig7b_public337_known_lattice.csv"
    write_csv(
        path,
        [
            "condition_key",
            "condition_label",
            "n",
            "protocol",
            "top1",
            "top1_ci_low",
            "top1_ci_high",
            "top20",
            "top20_ci_low",
            "top20_ci_high",
            "oracle",
            "small_n",
            "small_top1",
            "small_top20",
            "small_oracle",
        ],
        rows,
    )
    shutil.copy(path, PANEL_DIR / "fig7b_public337_known_lattice.csv")
    return path


def prepare_steps() -> Path:
    blob = load_json(SOURCES["public337"])
    rows = [
        {
            "step": "PXRD → +XRF",
            "lattice": "known",
            "metric": "Top-1",
            "delta": int(blob["step_pxrd_to_pxrd_xrf_lat1"]["top1"][0]),
            "ci_low": int(blob["step_pxrd_to_pxrd_xrf_lat1"]["top1"][1]),
            "ci_high": int(blob["step_pxrd_to_pxrd_xrf_lat1"]["top1"][2]),
        },
        {
            "step": "+XRF → +ND+XRF",
            "lattice": "known",
            "metric": "Top-1",
            "delta": int(blob["step_pxrd_xrf_to_pxrd_neutron_xrf_lat1"]["top1"][0]),
            "ci_low": int(blob["step_pxrd_xrf_to_pxrd_neutron_xrf_lat1"]["top1"][1]),
            "ci_high": int(blob["step_pxrd_xrf_to_pxrd_neutron_xrf_lat1"]["top1"][2]),
        },
    ]
    path = DATA_DIR / "fig7b_paired_steps.csv"
    write_csv(path, ["step", "lattice", "metric", "delta", "ci_low", "ci_high"], rows)
    shutil.copy(path, PANEL_DIR / "fig7b_paired_steps.csv")
    return path


def prepare_licoo2_ranks() -> Path:
    src = SOURCES["licoo2_ranks"]
    if not src.is_file():
        raise FileNotFoundError(src)
    rows: list[dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["lattice"] != "known":
                continue
            rows.append(
                {
                    "model": raw["model"],
                    "model_label": raw["model_label"],
                    "condition": raw["condition"],
                    "condition_label": raw["condition_label"],
                    "best_k": raw["best_k"],
                    "bucket": raw["bucket"],
                    "t1": raw["t1"],
                }
            )
    path = DATA_DIR / "fig7c_licoo2_known_ranks.csv"
    write_csv(
        path,
        [
            "model",
            "model_label",
            "condition",
            "condition_label",
            "best_k",
            "bucket",
            "t1",
        ],
        rows,
    )
    shutil.copy(path, PANEL_DIR / "fig7c_licoo2_known_ranks.csv")
    return path


def main() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        prepare_rps(),
        prepare_public337(),
        prepare_steps(),
        prepare_licoo2_ranks(),
    ]
    if not SOURCES["licoo2_spectra"].is_file():
        raise FileNotFoundError(SOURCES["licoo2_spectra"])
    shutil.copy(SOURCES["licoo2_spectra"], PANEL_DIR / "fig7c_licoo2_spectra.csv")
    write_csv(
        PANEL_DIR / "panel_manifest.csv",
        [
            "figure",
            "panel",
            "plot_title",
            "csv_path",
            "source_table",
            "row_count",
            "scope",
            "notes",
        ],
        [
            {
                "figure": "Fig. 7",
                "panel": "a",
                "plot_title": "CNRS/RRUFF versus RealPXRD-Solver",
                "csv_path": "fig7a_rps_head_to_head.csv",
                "source_table": "crystal_language/eval_results/chgnet/rps_*_bootstrap_refined.json",
                "row_count": 4,
                "scope": "CNRS n=126; RRUFF n=269; formula + known lattice; refine-before-match",
                "notes": "ours is CHGNet ∪ MACE refine; CIs are paired bootstrap on this work",
            },
            {
                "figure": "Fig. 7",
                "panel": "b",
                "plot_title": "Public 337 known-lattice spectroscopic ladder",
                "csv_path": "fig7b_public337_known_lattice.csv",
                "source_table": "crystal_language/eval_results/chgnet/ladder_public_si_report_spec_cm.json",
                "row_count": 5,
                "scope": "N=337 measured Cu Kα; ND/XRF CIF-simulated; no formula on spectroscopic rungs",
                "notes": "counts, not an accuracy rate; formula reference is unrefined fusion",
            },
            {
                "figure": "Fig. 7",
                "panel": "b",
                "plot_title": "Public 337 known-lattice paired steps",
                "csv_path": "fig7b_paired_steps.csv",
                "source_table": "crystal_language/eval_results/chgnet/ladder_public_si_report_spec_cm.json",
                "row_count": 2,
                "scope": "Paired Top-1 increments on the same 337 records",
                "notes": "+XRF → +ND+XRF interval includes 0",
            },
            {
                "figure": "Fig. 7",
                "panel": "c",
                "plot_title": "LiCoO2 known-lattice ranks",
                "csv_path": "fig7c_licoo2_known_ranks.csv",
                "source_table": "data/panels/fig7c_licoo2_ranks.csv",
                "row_count": 8,
                "scope": "One laboratory case; clean versus noisy-exp; known lattice",
                "notes": "Noisy PXRD is rank 2; +ND is Top-1; ND is CIF-simulated",
            },
            {
                "figure": "Fig. 7",
                "panel": "c",
                "plot_title": "LiCoO2 experimental PXRD versus simulated ND",
                "csv_path": "fig7c_licoo2_spectra.csv",
                "source_table": "data/licoo2_spectra.csv",
                "row_count": 510,
                "scope": "Keep-set LiCoO2; measured Bruker RAW vs CIF-simulated ND",
                "notes": "PXRD is experimental; ND is simulated from the paired CIF",
            },
        ],
    )
    for path in paths:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
