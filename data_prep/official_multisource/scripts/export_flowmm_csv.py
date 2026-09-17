#!/usr/bin/env python
"""Convert crystal-native JSONL exports into FlowMM/DiffCSP CSV splits.

FlowMM currently reuses DiffCSP's ``CrystDataset`` interface, which expects a
CSV with at least ``material_id``, ``pretty_formula``, ``cif`` and the target
property columns. Our project keeps richer PXRD/neutron/XRF/XPS conditions in
JSONL. This script produces both:

- CSV files that FlowMM/DiffCSP can read immediately;
- sidecar condition JSONL files keyed by ``material_id`` for the next
  spectrum-conditioned model extension.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure


DEFAULT_IN_DIR = Path("data_prep/outputs/crystal_native_unified")
DEFAULT_OUT_DIR = Path("data_prep/outputs/flowmm_unified")


def finite_or_default(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def structure_from_record(record: dict[str, Any]) -> Structure:
    return Structure(
        lattice=Lattice(np.asarray(record["lattice_matrix"], dtype=float)),
        species=list(record["atom_types"]),
        coords=np.asarray(record["frac_coords"], dtype=float),
        coords_are_cartesian=False,
    )


def row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    structure = structure_from_record(record)
    properties = record.get("properties") or {}
    material_id = record.get("id") or f"{record.get('source')}:{record.get('source_id')}"
    formula = record.get("formula") or structure.composition.reduced_formula
    return {
        "material_id": material_id,
        "pretty_formula": formula,
        "formula": formula,
        "source": record.get("source"),
        "source_id": record.get("source_id"),
        "num_sites": int(record.get("n_sites") or len(structure)),
        "spacegroup": record.get("space_group"),
        "formation_energy_per_atom": finite_or_default(
            properties.get("formation_energy_per_atom")
        ),
        "e_above_hull": finite_or_default(properties.get("e_above_hull")),
        "band_gap": finite_or_default(properties.get("band_gap")),
        "cif": structure.to(fmt="cif"),
    }


def condition_sidecar(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "material_id": record.get("id"),
        "split": record.get("split"),
        "conditions": record.get("conditions") or {},
        "properties": record.get("properties") or {},
        "lattice_matrix": record.get("lattice_matrix"),
        "formula": record.get("formula"),
        "n_sites": record.get("n_sites"),
    }


def convert_split(in_path: Path, out_csv: Path, out_conditions: Path) -> int:
    rows: list[dict[str, Any]] = []
    out_conditions.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as in_handle, out_conditions.open(
        "w", encoding="utf-8"
    ) as cond_handle:
        for line in in_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(row_from_record(record))
            cond_handle.write(json.dumps(condition_sidecar(record), ensure_ascii=False))
            cond_handle.write("\n")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits", default="train,valid,test")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    summary: dict[str, Any] = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "splits": {},
        "flowmm_data_config": "external_models/flowmm/scripts_model/conf/data/pxrd_mof.yaml",
    }

    for split in splits:
        in_path = in_dir / f"{split}.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(f"missing JSONL split: {in_path}")
        count = convert_split(
            in_path=in_path,
            out_csv=out_dir / f"{split}.csv",
            out_conditions=out_dir / "conditions" / f"{split}.jsonl",
        )
        summary["splits"][split] = count

    (out_dir / "flowmm_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
