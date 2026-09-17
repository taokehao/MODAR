#!/usr/bin/env python
"""Import the RealPXRD 6M LMDB (``pxrd_241113_*.lmdb``) into our crystal-native
JSONL schema with full multimodal enrichment.

Each LMDB value is gzip+pickle of a dict with::

    p_atom_type      list[str]        primitive cell element symbols
    p_atom_pos       (N, 3) float     fractional coordinates
    p_lattice_matrix (3, 3) float     lattice matrix (Angstrom)
    pxrd_x / pxrd_y  (24,) float      24 strongest peaks (kept as provenance)

We rebuild a pymatgen ``Structure`` and run the project's standard enrichment
(PXRD 5-80 deg CuKa, neutron pattern, XPS-style valences, XRF mass fractions)
so the downstream sequence exporter and V2 dataloader work unchanged and every
modality is simulated consistently with MP-20 / MP+WBM.

Sharding: ``--start-index`` / ``--max-records`` slice the integer LMDB keys so a
Slurm array can process the 6M split in parallel; concatenate shard outputs
afterwards.

Usage:
  python import_realpxrd.py --lmdb <path> --split train --out-dir <dir> \
      --start-index 0 --max-records 200000 --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))


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


def _decode(raw: bytes) -> dict[str, Any]:
    return pickle.loads(gzip.decompress(raw))


def convert_record(args: tuple[bytes, str, str]) -> str | None:
    """Worker: one LMDB value -> crystal-native JSONL line (None on failure)."""
    raw, split, key = args
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from pymatgen.core import Element, Lattice, Structure

        from src.structure_utils import structure_to_training_record

        rec = _decode(raw)
        atom_type = [str(s) for s in rec["p_atom_type"]]
        frac = np.mod(np.asarray(rec["p_atom_pos"], dtype=float).reshape(-1, 3), 1.0)
        lat = np.asarray(rec["p_lattice_matrix"], dtype=float).reshape(3, 3)
        structure = Structure(
            Lattice(lat), atom_type, frac, coords_are_cartesian=False, to_unit_cell=True
        )

        record = structure_to_training_record(
            structure=structure,
            source="realpxrd",
            source_id=key,
            formation_energy_per_atom=None,
            e_above_hull=None,
            band_gap=None,
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
            "id": f"realpxrd:{key}",
            "source": "realpxrd",
            "source_id": key,
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
                "formation_energy_per_atom": None,
                "e_above_hull": None,
                "band_gap": None,
                "bvse_li_e1d": None,
                "bvse_na_e1d": None,
            },
        }
        return json.dumps(to_jsonable(out), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return None


def iter_raw_values(lmdb_path: Path, start: int, count: int | None):
    import lmdb

    env = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, subdir=False, readahead=False
    )
    with env.begin() as txn:
        total = txn.stat()["entries"]
        end = total if count is None else min(total, start + count)
        for i in range(start, end):
            raw = txn.get(str(i).encode())
            if raw is not None:
                yield raw, str(i)
    env.close()


def import_shard(
    lmdb_path: Path,
    out_path: Path,
    split: str,
    start: int,
    count: int | None,
    workers: int,
) -> dict[str, Any]:
    t0 = time.time()
    stats: Counter[str] = Counter()
    jobs = ((raw, split, key) for raw, key in iter_raw_values(lmdb_path, start, count))
    with out_path.open("w", encoding="utf-8") as handle:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, line in enumerate(pool.map(convert_record, jobs, chunksize=16)):
                    if line is None:
                        stats["failed"] += 1
                    else:
                        handle.write(line + "\n")
                        stats["written"] += 1
                    if (i + 1) % 5000 == 0:
                        rate = (time.time() - t0) / (i + 1) * 1000
                        print(f"  {split}[{start}]: {i + 1} ({rate:.0f} ms/rec)", flush=True)
        else:
            for job in jobs:
                line = convert_record(job)
                if line is None:
                    stats["failed"] += 1
                else:
                    handle.write(line + "\n")
                    stats["written"] += 1
    stats["elapsed_sec"] = int(time.time() - t0)
    stats["total"] = stats["written"] + stats["failed"]
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmdb-dir",
        default="/home/tkh/python-project/alex_aflow_oqmd_mp/datasets",
        help="directory holding pxrd_241113_{train,valid,test}.lmdb",
    )
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data_prep" / "outputs" / "crystal_native_realpxrd"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--shard-tag",
        default=None,
        help="optional suffix for the shard output file (default: start index)",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="write {split}.jsonl (consumed by export_sequences) instead of a tagged shard",
    )
    args = parser.parse_args()

    lmdb_path = Path(args.lmdb_dir) / f"pxrd_241113_{args.split}.lmdb"
    if not lmdb_path.exists():
        print(f"LMDB not found: {lmdb_path}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.shard_tag if args.shard_tag is not None else f"start{args.start_index}"
    out_path = out_dir / (
        f"{args.split}.jsonl" if args.canonical else f"{args.split}_{tag}.jsonl"
    )

    print(
        f"importing {args.split} from {lmdb_path} "
        f"[start={args.start_index} max={args.max_records} workers={args.workers}]",
        flush=True,
    )
    stats = import_shard(
        lmdb_path, out_path, args.split, args.start_index, args.max_records, args.workers
    )
    print(json.dumps({f"{args.split}:{tag}": stats}, ensure_ascii=False), flush=True)
    summary_tag = "full" if args.canonical else tag
    (out_dir / f"{args.split}_{summary_tag}_summary.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
