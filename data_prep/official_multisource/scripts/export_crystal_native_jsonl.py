#!/usr/bin/env python
"""Export unified LMDB splits for crystal-native generator bake-offs.

The Uni-3DAR loader consumes compressed LMDB records and converts crystals into
grid/octree tokens. Crystal-native diffusion/flow models should instead start
from lattice matrices, fractional coordinates, atom types, and condition
signals. This script creates a JSONL bridge format without changing the source
LMDB files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element, Lattice
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import iter_lmdb  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "official_unified_multimodal"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "crystal_native_unified"

BAKEOFF_PROTOCOL: dict[str, Any] = {
    "baseline": {
        "name": "Uni-3DAR MP+WBM multimodal ablation",
        "checkpoint": "/home/tkh/ckps/wuji_yl_official_unified_20260521_171452/ckpt/checkpoint_best.pt",
        "strongest_current_condition": "A5",
        "a5_top1_match": "13/50",
        "a5_top10_match": "15/50",
    },
    "candidate_backbones": [
        {
            "name": "DiffCSP-style periodic diffusion",
            "role": "low-risk first crystal-native CSP/generation candidate",
        },
        {
            "name": "FlowMM-style crystal flow matching",
            "role": "second candidate with stronger future reward-weighted tuning path",
        },
    ],
    "condition_sets": {
        "B0": ["composition"],
        "B1": ["pxrd", "composition"],
        "B2": ["pxrd", "composition", "lattice"],
        "B3": ["pxrd", "neutron", "composition", "lattice"],
        "B4": ["pxrd", "neutron", "xrf", "xps_valence", "composition", "lattice"],
    },
    "metrics": [
        "valid_structure_rate",
        "composition_consistency",
        "structure_matcher_top1",
        "structure_matcher_topk",
        "matched_rms",
        "pxrd_similarity",
        "neutron_similarity",
        "duplicate_adjusted_topk_diversity",
    ],
    "promotion_criteria": [
        "beats Uni-3DAR A5 top-1 and top-10 match under the same split and sampling budget",
        "lattice-conditioned condition sets outperform lattice-free condition sets",
        "valid structure rate does not regress substantially",
        "conditioning path can support neutron, XRF, and XPS valence",
        "generated candidates can be passed into physics reward reranking",
    ],
}


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
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


def xrf_mass_fraction(atom_types: list[str]) -> dict[str, float]:
    masses: Counter[str] = Counter()
    for symbol in atom_types:
        try:
            masses[symbol] += float(Element(symbol).atomic_mass)
        except Exception:
            continue
    total = float(sum(masses.values()))
    if total <= 0:
        return {}
    return {symbol: mass / total for symbol, mass in sorted(masses.items())}


def lattice_parameters(lattice_matrix: np.ndarray) -> dict[str, list[float]]:
    lattice = Lattice(lattice_matrix)
    return {
        "lengths": [float(v) for v in lattice.abc],
        "angles": [float(v) for v in lattice.angles],
    }


def record_id(key: bytes, record: dict[str, Any]) -> str:
    source = record.get("source")
    source_id = record.get("source_id")
    if source and source_id:
        return f"{source}:{source_id}"
    return key.decode("utf-8", "ignore")


def convert_record(key: bytes, record: dict[str, Any], split: str) -> dict[str, Any]:
    atom_types = [str(x) for x in record["atom_type"]]
    frac_coords = np.mod(np.asarray(record["atom_pos"], dtype=float).reshape(-1, 3), 1.0)
    lattice_matrix = np.asarray(record["lattice_matrix"], dtype=float).reshape(3, 3)
    if len(atom_types) != frac_coords.shape[0]:
        raise ValueError(
            f"atom_type length {len(atom_types)} != frac_coords rows {frac_coords.shape[0]}"
        )

    conditions = {
        "pxrd_x": record.get("pxrd_x"),
        "pxrd_y": record.get("pxrd_y"),
        "neutron_x": record.get("neutron_x"),
        "neutron_y": record.get("neutron_y"),
        "xrf_mass_frac": xrf_mass_fraction(atom_types),
        "valence_assignments": record.get("valence_assignments") or {},
        "valence_source": record.get("valence_source"),
        "valence_assigned": bool(record.get("valence_assigned")),
    }

    properties = {
        "formation_energy_per_atom": finite_or_none(
            record.get("formation_energy_per_atom")
        ),
        "e_above_hull": finite_or_none(record.get("e_above_hull")),
        "band_gap": finite_or_none(record.get("band_gap")),
        "bvse_li_e1d": finite_or_none(record.get("bvse_li_e1d")),
        "bvse_na_e1d": finite_or_none(record.get("bvse_na_e1d")),
    }

    out = {
        "id": record_id(key, record),
        "source": record.get("source"),
        "source_id": record.get("source_id"),
        "split": split,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "lattice_matrix": lattice_matrix,
        "lattice_parameters": lattice_parameters(lattice_matrix),
        "formula": record.get("formula") or record.get("pretty_formula"),
        "n_sites": int(record.get("n_sites") or len(atom_types)),
        "space_group": record.get("space_group"),
        "structure_hash": record.get("structure_hash"),
        "conditions": conditions,
        "properties": properties,
    }
    return to_jsonable(out)


def export_split(
    lmdb_path: Path,
    out_path: Path,
    split: str,
    max_records: int | None,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as handle:
        iterator = iter_lmdb(str(lmdb_path), compressed=True)
        for i, (key, record) in enumerate(tqdm(iterator, desc=f"export {split}")):
            if max_records is not None and i >= max_records:
                break
            if keep_ids is not None and record_id(key, record) not in keep_ids:
                stats["filtered_out"] += 1
                continue
            try:
                converted = convert_record(key, record, split)
            except Exception as exc:
                stats["skipped"] += 1
                stats[f"skip_reason:{type(exc).__name__}"] += 1
                continue
            handle.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            stats["written"] += 1
            if converted["conditions"]["neutron_x"] is not None:
                stats["with_neutron"] += 1
            if converted["conditions"]["valence_assigned"]:
                stats["with_valence"] += 1
    return dict(stats)


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [split for split in splits if split not in {"train", "valid", "test"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid splits: {invalid}")
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits", type=parse_splits, default=parse_splits("train,valid,test"))
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        default=None,
        help="optional smoke-test cap for each split",
    )
    parser.add_argument(
        "--keep-ids",
        default=None,
        help="optional newline-delimited id allowlist (record_id = source:source_id); "
        "only matching records are exported (e.g. Tier-A dedup keep list)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_ids: set[str] | None = None
    if args.keep_ids:
        with open(args.keep_ids, encoding="utf-8") as handle:
            keep_ids = {line.strip() for line in handle if line.strip()}
        print(f"loaded {len(keep_ids)} keep ids from {args.keep_ids}", flush=True)

    summary: dict[str, Any] = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "max_records_per_split": args.max_records_per_split,
        "keep_ids_file": args.keep_ids,
        "n_keep_ids": (len(keep_ids) if keep_ids is not None else None),
        "splits": {},
        "protocol_file": "bakeoff_protocol.json",
    }

    for split in args.splits:
        lmdb_path = data_dir / f"{split}.lmdb"
        if not lmdb_path.exists():
            raise FileNotFoundError(f"missing split LMDB: {lmdb_path}")
        out_path = out_dir / f"{split}.jsonl"
        summary["splits"][split] = export_split(
            lmdb_path=lmdb_path,
            out_path=out_path,
            split=split,
            max_records=args.max_records_per_split,
            keep_ids=keep_ids,
        )

    (out_dir / "bakeoff_protocol.json").write_text(
        json.dumps(BAKEOFF_PROTOCOL, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
