#!/usr/bin/env python
"""Build Uni-3DAR-compatible LMDBs from AFLOW raw JSONL shard files.

The downloader (``01_download_extra_sources.py --download-aflow-catalog``)
emits one JSONL line per AFLOW entry containing fields like ``geometry``,
``positions_fractional``, ``species``, ``compound``, ``enthalpy_formation_atom``
and an optional ``structure_relax_json`` block.  This script parses those rows
into pymatgen ``Structure`` objects and writes the standard training record
schema (PXRD + neutron + valence + provenance + properties).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import LmdbWriter  # noqa: E402
from src.split_utils import split_from_key  # noqa: E402
from src.structure_utils import structure_to_training_record  # noqa: E402

DEFAULT_RAW = PROJECT_ROOT / "data_prep" / "raw" / "official_extra_sources"
DEFAULT_OUT = PROJECT_ROOT / "data_prep" / "outputs" / "official_aflow_props"

_COMPOUND_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def expand_compound_to_species(compound: str, expected_natoms: int | None = None) -> list[str]:
    """Expand AFLOW ``compound`` like ``"Ca1Nb3"`` into a per-site species list."""
    species: list[str] = []
    for el, count in _COMPOUND_RE.findall(compound or ""):
        if not el:
            continue
        n = int(count) if count else 1
        species.extend([el] * n)
    if expected_natoms is not None and len(species) != expected_natoms:
        return []
    return species


def _structure_from_relax_json(block: dict[str, Any]) -> Structure | None:
    """Parse the AFLOW ``_structure_relax.json`` block when present."""
    try:
        lattice = block.get("lattice") or block.get("lattice_vectors")
        atoms = block.get("atoms")
        if lattice is None or atoms is None:
            return None
        species = [a.get("name") or a.get("element") for a in atoms]
        positions = [a.get("position") or a.get("coord") for a in atoms]
        if any(s is None for s in species) or any(p is None for p in positions):
            return None
        return Structure(
            lattice=Lattice(np.asarray(lattice, dtype=float)),
            species=species,
            coords=positions,
            coords_are_cartesian=False,
        )
    except Exception:
        return None


def aflow_structure_from_row(row: dict[str, Any]) -> Structure | None:
    """Best-effort conversion of an AFLOW JSONL row to a pymatgen Structure."""
    block = row.get("structure_relax_json")
    if isinstance(block, dict):
        struct = _structure_from_relax_json(block)
        if struct is not None:
            return struct

    geometry = row.get("geometry")
    positions = row.get("positions_fractional")
    natoms = row.get("natoms")
    compound = row.get("compound")
    if not (isinstance(geometry, (list, tuple)) and len(geometry) == 6):
        return None
    if not (isinstance(positions, list) and positions):
        return None
    species = expand_compound_to_species(
        compound or "", expected_natoms=int(natoms) if natoms is not None else len(positions)
    )
    if not species or len(species) != len(positions):
        return None
    try:
        lattice = Lattice.from_parameters(*geometry)
        return Structure(
            lattice=lattice,
            species=species,
            coords=positions,
            coords_are_cartesian=False,
        )
    except Exception:
        return None


def iter_aflow_records(
    paths: Iterable[Path],
    max_records: int | None = None,
) -> Iterable[dict[str, Any]]:
    seen = 0
    for path in paths:
        with path.open() as f:
            for line in f:
                if max_records is not None and seen >= max_records:
                    return
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("download_error"):
                    continue
                structure = aflow_structure_from_row(row)
                if structure is None:
                    continue
                aurl = row.get("aurl") or row.get("input_aurl")
                source_id = row.get("auid") or aurl
                if not source_id:
                    continue
                yield structure_to_training_record(
                    structure=structure,
                    source="aflow",
                    source_id=str(source_id),
                    formation_energy_per_atom=finite_or_none(
                        row.get("enthalpy_formation_atom")
                    ),
                    e_above_hull=None,
                    band_gap=finite_or_none(row.get("Egap")),
                    multimodal=multimodal_from_row(row),
                    extra={
                        "aflow_aurl": aurl,
                        "aflow_compound": row.get("compound"),
                        "aflow_prototype": row.get("prototype"),
                        "aflow_spacegroup_relax": row.get("spacegroup_relax"),
                        "aflow_egap_type": row.get("Egap_type"),
                        "aflow_dft_type": row.get("dft_type"),
                        "aflow_code": row.get("code"),
                    },
                )
                seen += 1


def discover_shards(raw_dir: Path) -> list[Path]:
    enriched = sorted(raw_dir.glob("aflow_lib*_raw_shard_*_of_*.multimodal.jsonl"))
    if enriched:
        return [p for p in enriched if p.stat().st_size > 0]
    candidates = sorted(raw_dir.glob("aflow_lib*_raw_shard_*_of_*.jsonl"))
    return [p for p in candidates if p.stat().st_size > 0]


def multimodal_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    keys = ("neutron_x", "neutron_y", "valence_assignments", "valence_source", "valence_assigned")
    if not any(k in row for k in keys):
        return None
    return {k: row.get(k) for k in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--shard",
        action="append",
        default=None,
        help="explicit AFLOW shard JSONL path; can be repeated. If omitted, "
        "all aflow_lib*_raw_shard_*_of_*.jsonl in --raw-dir are scanned.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--valid-pct", type=float, default=0.01)
    parser.add_argument("--test-pct", type=float, default=0.01)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--map-size-gb", type=int, default=512)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shard:
        shards = [Path(s) for s in args.shard]
    else:
        shards = discover_shards(raw_dir)
    if not shards:
        raise SystemExit(f"No AFLOW shard JSONL files found under {raw_dir}.")
    print(f"AFLOW shards ({len(shards)}):")
    for p in shards:
        print(f"  {p}  ({p.stat().st_size/1024/1024:.1f} MB)")

    writers = {
        split: LmdbWriter(
            str(out_dir / f"{split}.lmdb"),
            map_size=args.map_size_gb * 1024**3,
            commit_every=1000,
            compress=True,
        )
        for split in ("train", "valid", "test")
    }

    counts: Counter = Counter()
    missing = defaultdict(Counter)
    seen_hashes: set[str] = set()
    try:
        for record in tqdm(
            iter_aflow_records(shards, max_records=args.max_records),
            desc="build aflow",
        ):
            if args.dedupe:
                h = record.get("structure_hash")
                if h in seen_hashes:
                    counts["aflow:deduped"] += 1
                    continue
                if h is not None:
                    seen_hashes.add(h)

            key = f"{record['source']}:{record['source_id']}"
            split = split_from_key(key, valid_pct=args.valid_pct, test_pct=args.test_pct)
            writers[split].put(record, key=key.encode("utf-8"))
            counts[f"aflow:{split}"] += 1
            counts[f"total:{split}"] += 1
            for prop in ("formation_energy_per_atom", "e_above_hull", "band_gap"):
                if record.get(prop) is None:
                    missing["aflow"][prop] += 1
            for aux in ("neutron_y", "valence_assigned"):
                value = record.get(aux)
                ok = bool(value) if isinstance(value, bool) else value is not None
                if not ok:
                    missing["aflow"][aux] += 1
    finally:
        for writer in writers.values():
            writer.close()

    report = {
        "counts": dict(counts),
        "missing_by_source": {k: dict(v) for k, v in missing.items()},
        "args": vars(args),
        "shards": [str(p) for p in shards],
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"LMDBs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
