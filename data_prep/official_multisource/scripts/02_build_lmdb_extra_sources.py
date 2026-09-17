#!/usr/bin/env python
"""Build Uni-3DAR-compatible LMDBs from Alexandria/OQMD/AFLOW raw JSONL files."""

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
DEFAULT_OUT = PROJECT_ROOT / "data_prep" / "outputs" / "official_extra_sources_debug"


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def load_manifest(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "download_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def iter_jsonl(
    path: Path,
    max_records: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterable[dict[str, Any]]:
    yielded = 0
    with path.open() as f:
        for i, line in enumerate(f):
            if i % shard_count != shard_index:
                continue
            if max_records is not None and yielded >= max_records:
                break
            yielded += 1
            yield json.loads(line)


def multimodal_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    keys = ("neutron_x", "neutron_y", "valence_assignments", "valence_source", "valence_assigned")
    if not any(k in row for k in keys):
        return None
    return {k: row.get(k) for k in keys}


def alexandria_structure_from_optimade(row: dict[str, Any]) -> Structure:
    attrs = row["attributes"]
    return Structure(
        lattice=attrs["lattice_vectors"],
        species=attrs["species_at_sites"],
        coords=attrs["cartesian_site_positions"],
        coords_are_cartesian=True,
    )


def iter_alexandria_records(
    path: Path,
    max_records: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterable[dict[str, Any]]:
    for row in iter_jsonl(path, max_records=max_records, shard_index=shard_index, shard_count=shard_count):
        attrs = row["attributes"]
        source_id = row.get("id") or attrs.get("immutable_id")
        structure = alexandria_structure_from_optimade(row)
        yield structure_to_training_record(
            structure=structure,
            source="alexandria",
            source_id=str(source_id),
            formation_energy_per_atom=attrs.get("_alexandria_formation_energy_per_atom"),
            e_above_hull=attrs.get("_alexandria_hull_distance"),
            band_gap=attrs.get("_alexandria_band_gap"),
            multimodal=multimodal_from_row(row),
            extra={
                "alexandria_dataset_id": attrs.get("immutable_id"),
                "alexandria_formula": attrs.get("chemical_formula_reduced"),
                "alexandria_xc_functional": attrs.get("_alexandria_xc_functional"),
                "alexandria_space_group": attrs.get("_alexandria_space_group"),
                "alexandria_band_gap_direct": finite_or_none(attrs.get("_alexandria_band_gap_direct")),
            },
        )


_SITE_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*@\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def oqmd_structure_from_row(row: dict[str, Any]) -> Structure:
    species: list[str] = []
    frac_coords: list[list[float]] = []
    for site in row["sites"]:
        match = _SITE_RE.match(site)
        if not match:
            raise ValueError(f"Cannot parse OQMD site string: {site!r}")
        species.append(match.group(1))
        frac_coords.append([float(match.group(i)) for i in (2, 3, 4)])
    return Structure(
        lattice=Lattice(row["unit_cell"]),
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def iter_oqmd_records(
    path: Path,
    max_records: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterable[dict[str, Any]]:
    for row in iter_jsonl(path, max_records=max_records, shard_index=shard_index, shard_count=shard_count):
        structure = oqmd_structure_from_row(row)
        source_id = f"oqmd:{row.get('entry_id')}:{row.get('calculation_id')}"
        yield structure_to_training_record(
            structure=structure,
            source="oqmd",
            source_id=source_id,
            formation_energy_per_atom=row.get("delta_e"),
            e_above_hull=row.get("stability"),
            band_gap=row.get("band_gap"),
            multimodal=multimodal_from_row(row),
            extra={
                "oqmd_entry_id": row.get("entry_id"),
                "oqmd_calculation_id": row.get("calculation_id"),
                "oqmd_formationenergy_id": row.get("formationenergy_id"),
                "oqmd_composition": row.get("composition"),
                "oqmd_prototype": row.get("prototype"),
                "oqmd_fit": row.get("fit"),
                "oqmd_calculation_label": row.get("calculation_label"),
                "oqmd_spacegroup": row.get("spacegroup"),
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--include-alexandria", action="store_true")
    parser.add_argument("--include-oqmd", action="store_true")
    parser.add_argument("--alexandria-jsonl", default=None)
    parser.add_argument("--oqmd-jsonl", default=None)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--valid-pct", type=float, default=0.01)
    parser.add_argument("--test-pct", type=float, default=0.01)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--map-size-gb", type=int, default=16)
    args = parser.parse_args(argv)

    if not args.include_alexandria and not args.include_oqmd:
        parser.error("Choose at least one of --include-alexandria / --include-oqmd")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)

    sources: list[tuple[str, Iterable[dict[str, Any]]]] = []
    if args.include_alexandria:
        path = Path(args.alexandria_jsonl or manifest.get("alexandria_pbe_jsonl", ""))
        enriched = path.with_name(path.name[: -len(".jsonl")] + ".multimodal.jsonl") if path.name.endswith(".jsonl") else path
        if args.alexandria_jsonl is None and enriched.exists():
            path = enriched
        if not path.exists():
            raise SystemExit("Alexandria JSONL not found. Run 01_download_extra_sources.py first.")
        sources.append((
            "alexandria",
            iter_alexandria_records(
                path,
                args.max_records_per_source,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ),
        ))

    if args.include_oqmd:
        path = Path(args.oqmd_jsonl or manifest.get("oqmd_formationenergy_jsonl", ""))
        enriched = path.with_name(path.name[: -len(".jsonl")] + ".multimodal.jsonl") if path.name.endswith(".jsonl") else path
        if args.oqmd_jsonl is None and enriched.exists():
            path = enriched
        if not path.exists():
            raise SystemExit("OQMD JSONL not found. Run 01_download_extra_sources.py first.")
        sources.append((
            "oqmd",
            iter_oqmd_records(
                path,
                args.max_records_per_source,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ),
        ))

    writers = {
        split: LmdbWriter(
            str(out_dir / f"{split}.lmdb"),
            map_size=args.map_size_gb * 1024**3,
            commit_every=1000,
            compress=True,
        )
        for split in ("train", "valid", "test")
    }

    counts = Counter()
    missing = defaultdict(Counter)
    seen_hashes: set[str] = set()
    try:
        for source, records in sources:
            for record in tqdm(records, desc=f"build {source}"):
                if args.dedupe:
                    structure_hash = record.get("structure_hash")
                    if structure_hash in seen_hashes:
                        counts[f"{source}:deduped"] += 1
                        continue
                    seen_hashes.add(structure_hash)

                key = f"{record['source']}:{record['source_id']}"
                split = split_from_key(key, valid_pct=args.valid_pct, test_pct=args.test_pct)
                writers[split].put(record, key=key.encode("utf-8"))
                counts[f"{source}:{split}"] += 1
                counts[f"total:{split}"] += 1
                for prop in ("formation_energy_per_atom", "e_above_hull", "band_gap"):
                    if record.get(prop) is None:
                        missing[source][prop] += 1
    finally:
        for writer in writers.values():
            writer.close()

    report = {
        "counts": dict(counts),
        "missing_by_source": {k: dict(v) for k, v in missing.items()},
        "args": vars(args),
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"LMDBs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

