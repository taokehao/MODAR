#!/usr/bin/env python
"""Build Uni-3DAR-compatible LMDBs from MP + WBM official data.

The output schema intentionally keeps the model-facing fields compatible with
the existing alex/mp20 scripts:
  atom_type, atom_pos, lattice_matrix, pxrd_x, pxrd_y

Additional metadata/property fields are appended:
  source, source_id, formation_energy_per_atom, e_above_hull, band_gap, ...
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import LmdbWriter  # noqa: E402
from src.split_utils import split_from_key  # noqa: E402
from src.structure_utils import structure_to_training_record  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "data_prep" / "outputs" / "official_mp_wbm_props"
DEFAULT_RAW = PROJECT_ROOT / "data_prep" / "raw" / "official_mp_wbm"


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def iter_mp_records(mp_jsonl: Path, max_records: int | None = None) -> Iterable[dict[str, Any]]:
    with mp_jsonl.open() as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            doc = json.loads(line)
            structure = Structure.from_dict(doc["structure"])
            source_id = str(doc.get("material_id"))
            yield structure_to_training_record(
                structure=structure,
                source="mp",
                source_id=source_id,
                formation_energy_per_atom=doc.get("formation_energy_per_atom"),
                e_above_hull=doc.get("energy_above_hull"),
                band_gap=doc.get("band_gap"),
                extra={
                    "mp_formula_pretty": doc.get("formula_pretty"),
                    "mp_nsites": doc.get("nsites"),
                },
            )


def _read_wbm_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Matbench Discovery has renamed/expanded columns across releases. Prefer
    # corrected values, falling back to original WBM columns.
    if "e_form_per_atom_mp2020_corrected" in df:
        df["formation_energy_per_atom_std"] = df["e_form_per_atom_mp2020_corrected"]
    elif "e_form_per_atom_wbm" in df:
        df["formation_energy_per_atom_std"] = df["e_form_per_atom_wbm"]
    else:
        raise ValueError("WBM summary lacks a formation energy column")

    if "e_above_hull_mp2020_corrected_ppd_mp" in df:
        df["e_above_hull_std"] = df["e_above_hull_mp2020_corrected_ppd_mp"]
    elif "e_above_hull_wbm" in df:
        df["e_above_hull_std"] = df["e_above_hull_wbm"]
    else:
        raise ValueError("WBM summary lacks an e_above_hull column")

    if "bandgap_pbe" in df:
        df["band_gap_std"] = df["bandgap_pbe"]
    elif "band_gap" in df:
        df["band_gap_std"] = df["band_gap"]
    else:
        df["band_gap_std"] = np.nan
    return df.set_index("material_id", drop=False)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt")
    return open(path, "rt")


def _iter_wbm_initial_structures(path: Path) -> Iterable[tuple[str, Structure]]:
    # JSONL rows contain pymatgen Structure dicts.
    with _open_text(path) as f:
        for line in f:
            row = json.loads(line)
            mid = row.get("material_id")
            struct_dict = row.get("initial_structure") or row.get("structure")
            if mid is None or struct_dict is None:
                continue
            yield str(mid), Structure.from_dict(struct_dict)


def _iter_wbm_entries(path: Path) -> Iterable[tuple[str, Structure]]:
    # Fallback if the installed Matbench Discovery release exposes computed
    # entries but not initial structures.
    with _open_text(path) as f:
        for line in f:
            entry = ComputedStructureEntry.from_dict(json.loads(line))
            yield str(entry.entry_id), entry.structure


def iter_wbm_records(
    summary_csv: Path,
    structures_path: Path | None,
    entries_path: Path | None = None,
    max_records: int | None = None,
) -> Iterable[dict[str, Any]]:
    summary = _read_wbm_summary(summary_csv)
    if structures_path is not None and structures_path.exists():
        iterator = _iter_wbm_initial_structures(structures_path)
    elif entries_path is not None and entries_path.exists():
        iterator = _iter_wbm_entries(entries_path)
    else:
        raise FileNotFoundError("No WBM structures or computed entries file found")
    for i, (mid, structure) in enumerate(iterator):
        if max_records is not None and i >= max_records:
            break
        if mid not in summary.index:
            continue
        row = summary.loc[mid]
        yield structure_to_training_record(
            structure=structure,
            source="wbm",
            source_id=mid,
            formation_energy_per_atom=finite_or_none(row.get("formation_energy_per_atom_std")),
            e_above_hull=finite_or_none(row.get("e_above_hull_std")),
            band_gap=finite_or_none(row.get("band_gap_std")),
            extra={
                "wbm_formula": row.get("formula"),
                "wbm_n_sites": finite_or_none(row.get("n_sites")),
            },
        )


def load_download_manifest(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "download_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--include-mp", action="store_true")
    parser.add_argument("--include-wbm", action="store_true")
    parser.add_argument("--mp-jsonl", default=None)
    parser.add_argument("--wbm-summary", default=None)
    parser.add_argument("--wbm-structures", default=None)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--valid-pct", type=float, default=0.01)
    parser.add_argument("--test-pct", type=float, default=0.01)
    parser.add_argument("--dedupe", action="store_true", help="dedupe by primitive structure hash")
    parser.add_argument("--map-size-gb", type=int, default=256)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_download_manifest(raw_dir)

    if not args.include_mp and not args.include_wbm:
        parser.error("Choose at least one of --include-mp / --include-wbm")

    sources: list[tuple[str, Iterable[dict[str, Any]]]] = []
    if args.include_mp:
        mp_jsonl = Path(args.mp_jsonl or manifest.get("mp_summary_jsonl", ""))
        if not mp_jsonl.exists():
            raise SystemExit("MP JSONL not found. Run 01_download_mp_wbm.py --download-mp first or pass --mp-jsonl.")
        sources.append(("mp", iter_mp_records(mp_jsonl, args.max_records_per_source)))

    if args.include_wbm:
        wbm_info = manifest.get("wbm", {})
        summary = Path(args.wbm_summary or wbm_info.get("wbm_summary", ""))
        structures_val = args.wbm_structures or wbm_info.get("wbm_initial_structures")
        entries_val = wbm_info.get("wbm_computed_structure_entries")
        structures = Path(structures_val) if structures_val else None
        entries = Path(entries_val) if entries_val else None
        if not summary.exists() or not ((structures and structures.exists()) or (entries and entries.exists())):
            raise SystemExit(
                "WBM files not found. Run 01_download_mp_wbm.py --download-wbm first "
                "or pass --wbm-summary and --wbm-structures."
            )
        sources.append(("wbm", iter_wbm_records(summary, structures, entries, args.max_records_per_source)))

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
                    h = record.get("structure_hash")
                    if h in seen_hashes:
                        counts[f"{source}:deduped"] += 1
                        continue
                    seen_hashes.add(h)

                key = f"{record['source']}:{record['source_id']}"
                split = split_from_key(key, valid_pct=args.valid_pct, test_pct=args.test_pct)
                writers[split].put(record)
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
