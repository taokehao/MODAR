#!/usr/bin/env python
"""Add neutron diffraction and XPS-style valence fields to an existing LMDB.

This is useful for datasets that have already been built, such as MP+WBM.  The
script reads train/valid/test LMDBs, reconstructs pymatgen Structures from the
stored primitive cell fields, computes neutron diffraction + oxidation-state
metadata, and writes a new LMDB directory without touching the input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import LmdbWriter, iter_lmdb  # noqa: E402
from src.multimodal_utils import compute_multimodal_signals  # noqa: E402

DEFAULT_IN = PROJECT_ROOT / "data_prep" / "outputs" / "official_mp_wbm_props"
DEFAULT_OUT = PROJECT_ROOT / "data_prep" / "outputs" / "official_mp_wbm_multimodal"


def structure_from_record(record: dict[str, Any]) -> Structure:
    return Structure(
        lattice=Lattice(np.asarray(record["lattice_matrix"], dtype=float)),
        species=list(record["atom_type"]),
        coords=np.asarray(record["atom_pos"], dtype=float),
        coords_are_cartesian=False,
    )


def has_multimodal_fields(record: dict[str, Any]) -> bool:
    return (
        record.get("neutron_x") is not None
        and record.get("neutron_y") is not None
        and record.get("valence_assignments") is not None
    )


def enrich_record(record: dict[str, Any], two_theta_min: float, two_theta_max: float) -> dict[str, Any]:
    if has_multimodal_fields(record):
        return record
    structure = structure_from_record(record)
    mm = compute_multimodal_signals(
        structure,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
        enable_neutron=True,
        enable_valence=True,
    )
    record = dict(record)
    record.update(mm)
    return record


def enrich_split(
    *,
    in_path: Path,
    out_path: Path,
    map_size_gb: int,
    two_theta_min: float,
    two_theta_max: float,
    max_records: int | None,
) -> dict[str, int]:
    counts: Counter = Counter()
    with LmdbWriter(
        str(out_path),
        map_size=map_size_gb * 1024**3,
        commit_every=1000,
        compress=True,
    ) as writer:
        for i, (key, record) in enumerate(tqdm(iter_lmdb(str(in_path), compressed=True), desc=in_path.stem)):
            if max_records is not None and i >= max_records:
                break
            try:
                enriched = enrich_record(record, two_theta_min, two_theta_max)
            except Exception as exc:
                enriched = dict(record)
                enriched["neutron_x"] = None
                enriched["neutron_y"] = None
                enriched["valence_assignments"] = None
                enriched["valence_source"] = "failed"
                enriched["valence_assigned"] = False
                enriched["multimodal_error"] = repr(exc)
                counts["failed"] += 1
            if enriched.get("neutron_y") is None:
                counts["missing_neutron"] += 1
            if not enriched.get("valence_assigned"):
                counts["missing_valence"] += 1
            writer.put(enriched, key=key)
            counts["written"] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=str(DEFAULT_IN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--map-size-gb", type=int, default=256)
    parser.add_argument("--two-theta-min", type=float, default=5.0)
    parser.add_argument("--two-theta-max", type=float, default=80.0)
    parser.add_argument("--max-records-per-split", type=int, default=None)
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"args": vars(args), "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        in_path = in_dir / f"{split}.lmdb"
        out_path = out_dir / f"{split}.lmdb"
        if not in_path.exists():
            print(f"skip missing split: {in_path}")
            continue
        print(f"enrich {in_path} -> {out_path}")
        report["splits"][split] = enrich_split(
            in_path=in_path,
            out_path=out_path,
            map_size_gb=args.map_size_gb,
            two_theta_min=args.two_theta_min,
            two_theta_max=args.two_theta_max,
            max_records=args.max_records_per_split,
        )

    (out_dir / "multimodal_enrich_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"enriched LMDB -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
