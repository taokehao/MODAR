#!/usr/bin/env python3
"""Prepare the curated source-data tables for manuscript Figure 6."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = FIGURE_DIR / "data"
RESULTS_DIR = REPO_ROOT / "crystal_language" / "eval_results"

EXPECTED_N = 500
HIGH_INFORMATION = ("pxrd_xrf", "pxrd_neutron_xrf", "pxrd_formula")
BASE_REFINEMENT_ORDER = (
    "pxrd_xrf",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
)
CONDITION_LABELS = {
    "pxrd_xrf": "PXRD + XRF",
    "pxrd_neutron_xrf": "PXRD + ND + XRF",
    "pxrd_neutron_xrf_xps": "PXRD + ND + XRF + XPS",
    "pxrd_formula": "Formula reference",
}

SOURCES = {
    "summary": RESULTS_DIR / "noisy_robustness_summary.json",
    "deploy": RESULTS_DIR / "noisy_rerank_deploy_base_ladder.json",
    "refine_base": RESULTS_DIR / "refine_rerank_ablation_base_free.json",
    "refine_bigcell": RESULTS_DIR / "refine_rerank_ablation_bigcell_free.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required source result is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_n(value: Any, context: str) -> int:
    n = int(value)
    if n != EXPECTED_N:
        raise ValueError(f"{context}: expected N={EXPECTED_N}, found N={n}")
    return n


def require_close(observed: float, expected: float, context: str) -> None:
    if abs(float(observed) - float(expected)) > 1e-8:
        raise ValueError(
            f"{context}: inconsistent delta ({observed} != {expected})"
        )


def write_csv(
    filename: str, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def prepare_clean_noisy(summary: dict[str, Any]) -> Path:
    rows: list[dict[str, Any]] = []
    for lattice_key, lattice_label in (
        ("free", "Lattice-free"),
        ("known", "Known lattice"),
    ):
        indexed = {
            entry["rung"]: entry for entry in summary["ladder"][lattice_key]
        }
        for condition in HIGH_INFORMATION:
            entry = indexed[condition]
            n_clean = require_n(
                entry["N_clean"], f"clean/noisy {lattice_key} {condition} clean"
            )
            n_noisy = require_n(
                entry["N_noisy"], f"clean/noisy {lattice_key} {condition} noisy"
            )
            require_close(
                entry["delta_T1"],
                float(entry["noisy_T1"]) - float(entry["clean_T1"]),
                f"clean/noisy {lattice_key} {condition} T1",
            )
            require_close(
                entry["delta_T20"],
                float(entry["noisy_T20"]) - float(entry["clean_T20"]),
                f"clean/noisy {lattice_key} {condition} T20",
            )
            rows.append(
                {
                    "pool_id": "noisy_robustness_summary_ladder",
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "lattice": lattice_key,
                    "lattice_label": lattice_label,
                    "N_clean": n_clean,
                    "N_noisy": n_noisy,
                    "clean_T1": f"{float(entry['clean_T1']):.1f}",
                    "noisy_T1": f"{float(entry['noisy_T1']):.1f}",
                    "delta_T1": f"{float(entry['delta_T1']):.1f}",
                    "clean_T20": f"{float(entry['clean_T20']):.1f}",
                    "noisy_T20": f"{float(entry['noisy_T20']):.1f}",
                    "delta_T20": f"{float(entry['delta_T20']):.1f}",
                }
            )
    return write_csv(
        "clean_noisy_high_information.csv",
        [
            "pool_id",
            "condition",
            "condition_label",
            "lattice",
            "lattice_label",
            "N_clean",
            "N_noisy",
            "clean_T1",
            "noisy_T1",
            "delta_T1",
            "clean_T20",
            "noisy_T20",
            "delta_T20",
        ],
        rows,
    )


def prepare_deploy(deploy: dict[str, Any]) -> Path:
    rows: list[dict[str, Any]] = []
    free_rows = [
        row for row in deploy["table"] if row["lattice"] == "lattice_free"
    ]
    for condition in HIGH_INFORMATION:
        condition_rows = [
            row for row in free_rows if row["condition"] == condition
        ]
        baseline_rows = [
            row
            for row in condition_rows
            if row["policy"].startswith("baseline peaks")
        ]
        deploy_rows = [
            row
            for row in condition_rows
            if row["policy"].startswith("deploy peaks")
        ]
        if len(baseline_rows) != 1 or len(deploy_rows) != 1:
            raise ValueError(
                f"Expected one baseline and one deploy row for {condition}"
            )
        baseline = baseline_rows[0]
        deployed = deploy_rows[0]
        require_n(baseline["N"], f"deploy baseline {condition}")
        require_n(deployed["N"], f"deploy policy {condition}")
        delta_t1 = float(deployed["T1"]) - float(baseline["T1"])
        delta_t20 = float(deployed["T20"]) - float(baseline["T20"])
        rows.append(
            {
                "pool_id": "noisy_base_ladder_frozen_candidates",
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "lattice": "free",
                "N": EXPECTED_N,
                "baseline_policy": baseline["policy"],
                "baseline_sigma": f"{float(baseline['params']['sigma']):.2f}",
                "baseline_shift_max": (
                    f"{float(baseline['params']['shift_max']):.2f}"
                ),
                "baseline_T1": f"{float(baseline['T1']):.1f}",
                "baseline_T20": f"{float(baseline['T20']):.1f}",
                "deploy_policy": deployed["policy"],
                "deploy_sigma": f"{float(deployed['params']['sigma']):.2f}",
                "deploy_shift_max": (
                    f"{float(deployed['params']['shift_max']):.2f}"
                ),
                "deploy_T1": f"{float(deployed['T1']):.1f}",
                "deploy_T20": f"{float(deployed['T20']):.1f}",
                "delta_T1": f"{delta_t1:.1f}",
                "delta_T20": f"{delta_t20:.1f}",
            }
        )
    return write_csv(
        "deploy_rerank_free.csv",
        [
            "pool_id",
            "condition",
            "condition_label",
            "lattice",
            "N",
            "baseline_policy",
            "baseline_sigma",
            "baseline_shift_max",
            "baseline_T1",
            "baseline_T20",
            "deploy_policy",
            "deploy_sigma",
            "deploy_shift_max",
            "deploy_T1",
            "deploy_T20",
            "delta_T1",
            "delta_T20",
        ],
        rows,
    )


def refinement_rows(
    payload: dict[str, Any], *, population: str, pool_id: str
) -> list[dict[str, Any]]:
    indexed = {entry["condition"]: entry for entry in payload["results"]}
    order = (
        BASE_REFINEMENT_ORDER
        if population == "base"
        else ("pxrd_neutron_xrf",)
    )
    rows: list[dict[str, Any]] = []
    for condition in order:
        entry = indexed[condition]
        if bool(entry["use_lattice"]):
            raise ValueError(f"Refinement row unexpectedly uses lattice: {condition}")
        baseline = entry["baseline_deploy"]
        refined = entry["refine_top_m"]
        require_n(baseline["N"], f"refinement baseline {population} {condition}")
        require_n(refined["N"], f"refinement refined {population} {condition}")
        delta_t1 = float(refined["T1"]) - float(baseline["T1"])
        delta_t20 = float(refined["T20"]) - float(baseline["T20"])
        require_close(
            entry["delta_T1"],
            delta_t1,
            f"refinement {population} {condition} T1",
        )
        rows.append(
            {
                "pool_id": pool_id,
                "population": population,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "lattice": "free",
                "N": EXPECTED_N,
                "baseline_T1": f"{float(baseline['T1']):.1f}",
                "refined_T1": f"{float(refined['T1']):.1f}",
                "delta_T1": f"{delta_t1:.1f}",
                "baseline_T20": f"{float(baseline['T20']):.1f}",
                "refined_T20": f"{float(refined['T20']):.1f}",
                "delta_T20": f"{delta_t20:.1f}",
                "refine_top_m": int(entry["refine_top_m_param"]),
            }
        )
    return rows


def prepare_refinement(
    refine_base: dict[str, Any], refine_bigcell: dict[str, Any]
) -> Path:
    rows = refinement_rows(
        refine_base,
        population="base",
        pool_id="refine_base_free_frozen_candidates",
    )
    rows.extend(
        refinement_rows(
            refine_bigcell,
            population="bigcell",
            pool_id="refine_bigcell_free_frozen_candidates",
        )
    )
    return write_csv(
        "refinement_free.csv",
        [
            "pool_id",
            "population",
            "condition",
            "condition_label",
            "lattice",
            "N",
            "baseline_T1",
            "refined_T1",
            "delta_T1",
            "baseline_T20",
            "refined_T20",
            "delta_T20",
            "refine_top_m",
        ],
        rows,
    )


def prepare_bigcell(summary: dict[str, Any]) -> Path:
    rows: list[dict[str, Any]] = []
    for lattice_key, lattice_label in (
        ("free", "Lattice-free"),
        ("known", "Known lattice"),
    ):
        entry = summary["bigcell"][lattice_key]
        require_close(
            entry["noisy_vs_clean_bigcell_T1"],
            float(entry["noisy_bigcell_T1"])
            - float(entry["clean_bigcell_T1"]),
            f"big-cell {lattice_key} T1",
        )
        require_close(
            entry["noisy_vs_clean_bigcell_T20"],
            float(entry["noisy_bigcell_T20"])
            - float(entry["clean_bigcell_T20"]),
            f"big-cell {lattice_key} T20",
        )
        rows.append(
            {
                "pool_id": "noisy_robustness_summary_bigcell",
                "condition": "pxrd_neutron_xrf",
                "condition_label": CONDITION_LABELS["pxrd_neutron_xrf"],
                "lattice": lattice_key,
                "lattice_label": lattice_label,
                "N": EXPECTED_N,
                "clean_base_T1": f"{float(entry['clean_base_T1']):.1f}",
                "clean_bigcell_T1": (
                    f"{float(entry['clean_bigcell_T1']):.1f}"
                ),
                "noisy_bigcell_T1": (
                    f"{float(entry['noisy_bigcell_T1']):.1f}"
                ),
                "delta_T1": (
                    f"{float(entry['noisy_vs_clean_bigcell_T1']):.1f}"
                ),
                "clean_bigcell_T20": (
                    f"{float(entry['clean_bigcell_T20']):.1f}"
                ),
                "noisy_bigcell_T20": (
                    f"{float(entry['noisy_bigcell_T20']):.1f}"
                ),
                "delta_T20": (
                    f"{float(entry['noisy_vs_clean_bigcell_T20']):.1f}"
                ),
            }
        )
    return write_csv(
        "bigcell_clean_noisy.csv",
        [
            "pool_id",
            "condition",
            "condition_label",
            "lattice",
            "lattice_label",
            "N",
            "clean_base_T1",
            "clean_bigcell_T1",
            "noisy_bigcell_T1",
            "delta_T1",
            "clean_bigcell_T20",
            "noisy_bigcell_T20",
            "delta_T20",
        ],
        rows,
    )


def main() -> None:
    payloads = {name: load_json(path) for name, path in SOURCES.items()}
    outputs = [
        prepare_clean_noisy(payloads["summary"]),
        prepare_deploy(payloads["deploy"]),
        prepare_refinement(
            payloads["refine_base"], payloads["refine_bigcell"]
        ),
        prepare_bigcell(payloads["summary"]),
    ]
    for path in outputs:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
