#!/usr/bin/env python
"""Import the official CDVAE MP-20 split and enrich it with all modalities.

Downloads the canonical CSVs (train 27,136 / val 9,047 / test 9,046) from the
CDVAE repository, parses each CIF, reduces to the primitive standard cell, and
runs the project's standard enrichment (PXRD 5-80 deg CuKa, neutron pattern,
XPS-style valence assignment). Output uses the crystal-native JSONL schema so
the downstream sequence exporter works unchanged.

Usage:
  python import_mp20.py --out-dir data_prep/outputs/crystal_native_mp20 [--workers 32]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

CSV_BASE = "https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20"
SPLIT_FILES = {"train": "train.csv", "valid": "val.csv", "test": "test.csv"}
DEFAULT_RAW_DIR = PROJECT_ROOT / "data_prep" / "raw" / "mp_20"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "crystal_native_mp20"

csv.field_size_limit(10_000_000)


def download_split(raw_dir: Path, split: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = SPLIT_FILES[split]
    dest = raw_dir / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = f"{CSV_BASE}/{filename}"
    print(f"downloading {url}", flush=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:
            print(f"  attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"failed to download {url}")


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def convert_row(args: tuple[dict[str, str], str]) -> str | None:
    """Worker: CSV row -> crystal-native JSONL line (None on failure)."""
    row, split = args
    warnings.filterwarnings("ignore")
    try:
        from pymatgen.core import Element, Lattice, Structure

        from src.structure_utils import structure_to_training_record

        structure = Structure.from_str(row["cif"], fmt="cif")
        record = structure_to_training_record(
            structure=structure,
            source="mp20",
            source_id=row["material_id"],
            formation_energy_per_atom=_finite_or_none(row.get("formation_energy_per_atom")),
            e_above_hull=_finite_or_none(row.get("e_above_hull")),
            band_gap=_finite_or_none(row.get("band_gap")),
        )

        atom_types = [str(x) for x in record["atom_type"]]
        frac_coords = np.mod(np.asarray(record["atom_pos"], dtype=float).reshape(-1, 3), 1.0)
        lattice_matrix = np.asarray(record["lattice_matrix"], dtype=float).reshape(3, 3)
        lattice = Lattice(lattice_matrix)

        masses: Counter[str] = Counter()
        for symbol in atom_types:
            masses[symbol] += float(Element(symbol).atomic_mass)
        total_mass = float(sum(masses.values()))
        xrf = (
            {symbol: mass / total_mass for symbol, mass in sorted(masses.items())}
            if total_mass > 0
            else {}
        )

        out = {
            "id": f"mp20:{row['material_id']}",
            "source": "mp20",
            "source_id": row["material_id"],
            "split": split,
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattice_matrix": lattice_matrix,
            "lattice_parameters": {
                "lengths": [float(v) for v in lattice.abc],
                "angles": [float(v) for v in lattice.angles],
            },
            "formula": record.get("formula"),
            "n_sites": int(record.get("n_sites") or len(atom_types)),
            "space_group": record.get("space_group"),
            "structure_hash": record.get("structure_hash"),
            "conditions": {
                "pxrd_x": record.get("pxrd_x"),
                "pxrd_y": record.get("pxrd_y"),
                "neutron_x": record.get("neutron_x"),
                "neutron_y": record.get("neutron_y"),
                "xrf_mass_frac": xrf,
                "valence_assignments": record.get("valence_assignments") or {},
                "valence_source": record.get("valence_source"),
                "valence_assigned": bool(record.get("valence_assigned")),
            },
            "properties": {
                "formation_energy_per_atom": record.get("formation_energy_per_atom"),
                "e_above_hull": record.get("e_above_hull"),
                "band_gap": record.get("band_gap"),
                "bvse_li_e1d": None,
                "bvse_na_e1d": None,
            },
        }
        return json.dumps(to_jsonable(out), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return None


def import_split(
    raw_csv: Path, out_path: Path, split: str, workers: int, max_records: int | None
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if max_records is not None and len(rows) >= max_records:
                break

    t0 = time.time()
    stats: Counter[str] = Counter(total=len(rows))
    jobs = [(row, split) for row in rows]
    with out_path.open("w", encoding="utf-8") as handle:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, line in enumerate(pool.map(convert_row, jobs, chunksize=16)):
                    if line is None:
                        stats["failed"] += 1
                    else:
                        handle.write(line + "\n")
                        stats["written"] += 1
                    if (i + 1) % 2000 == 0:
                        print(
                            f"  {split}: {i + 1}/{len(rows)} "
                            f"({(time.time() - t0) / (i + 1) * 1000:.0f} ms/rec)",
                            flush=True,
                        )
        else:
            for job in jobs:
                line = convert_row(job)
                if line is None:
                    stats["failed"] += 1
                else:
                    handle.write(line + "\n")
                    stats["written"] += 1
    stats["elapsed_sec"] = int(time.time() - t0)
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-records-per-split", type=int, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"raw_dir": str(raw_dir), "out_dir": str(out_dir), "splits": {}}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        raw_csv = download_split(raw_dir, split)
        out_path = out_dir / f"{split}.jsonl"
        print(f"importing {split} from {raw_csv}", flush=True)
        stats = import_split(
            raw_csv, out_path, split, args.workers, args.max_records_per_split
        )
        summary["splits"][split] = stats
        print(json.dumps({split: stats}, ensure_ascii=False), flush=True)

    (out_dir / "import_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
