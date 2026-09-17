#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import iter_lmdb  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "data_prep" / "outputs" / "official_mp_wbm_props"

PROPS = [
    ("formation_energy", "formation_energy_per_atom", lambda x: x, "identity"),
    ("band_gap", "band_gap", lambda x: x, "identity"),
    ("e_above_hull", "e_above_hull", lambda x: np.log1p(x), "log1p"),
]


def stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p01": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--split", default="train")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    lmdb_path = out_dir / f"{args.split}.lmdb"
    raw = {name: [] for name, *_ in PROPS}
    by_source: dict[str, dict[str, list[float]]] = {}
    n_total = 0
    source_counts: dict[str, int] = {}

    for _, rec in iter_lmdb(str(lmdb_path), compressed=True):
        n_total += 1
        source = rec.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        by_source.setdefault(source, {name: [] for name, *_ in PROPS})
        for name, key, _, _ in PROPS:
            v = rec.get(key)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            raw[name].append(v)
            by_source[source][name].append(v)

    result = {
        "n_total": n_total,
        "source_counts": source_counts,
        "props": {},
        "props_by_source": {},
    }
    for name, key, transform, transform_name in PROPS:
        arr = np.asarray(raw[name], dtype=np.float64)
        result["props"][name] = {
            "lmdb_key": key,
            "transform": transform_name,
            "raw_stats": stats(arr),
            "transformed_stats": stats(transform(arr)),
            "missing_count": n_total - int(arr.size),
            "missing_rate": (n_total - int(arr.size)) / max(n_total, 1),
        }
    for source, source_raw in by_source.items():
        result["props_by_source"][source] = {}
        n_source = source_counts[source]
        for name, key, transform, transform_name in PROPS:
            arr = np.asarray(source_raw[name], dtype=np.float64)
            result["props_by_source"][source][name] = {
                "lmdb_key": key,
                "transform": transform_name,
                "raw_stats": stats(arr),
                "transformed_stats": stats(transform(arr)),
                "missing_count": n_source - int(arr.size),
                "missing_rate": (n_source - int(arr.size)) / max(n_source, 1),
            }

    path = out_dir / "normalization_stats.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"stats -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
