#!/usr/bin/env python3
"""Carve independent temperature-selection sets matched to the SI size regime.

The sampling temperature was swept on CNRS/RRUFF themselves, so the winning
setting cannot be quoted without selecting it somewhere else first. Coverage
collapses with cell size, so the selection set has to sit in the same n_sites
band as the SI benchmarks (all <= 20 sites, median 12) rather than the full
public set (median 40).

Two disjoint-from-RPS sources are emitted:
  public_nle20  experimental opXRD/COD patterns, n_sites <= 20
  tierA_nle20   held-out Tier-A test records, n_sites <= 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
OUT_ROOT = ROOT / "data_prep/outputs/rps_tempselect"


def carve(
    src: Path,
    dst: Path,
    max_sites: int,
    limit: int,
    min_sites: int = 0,
    require_lattice: bool = True,
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            row = json.loads(line)
            n = row.get("n_sites")
            if n is None or int(n) > max_sites or int(n) < min_sites:
                continue
            cond = row.get("cond") or {}
            if require_lattice and not cond.get("lattice"):
                continue
            if not cond.get("formula"):
                continue
            spec = row.get("spectra") or {}
            if not spec.get("pxrd_x"):
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if limit and kept >= limit:
                break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sites", type=int, default=20)
    ap.add_argument("--public-limit", type=int, default=0)
    ap.add_argument("--tiera-limit", type=int, default=192)
    args = ap.parse_args()

    jobs = [
        (
            "public_nle20",
            ROOT / "data_prep/outputs/public_exp_pxrd/crystal_language_v2/test.jsonl",
            args.public_limit,
            0,
        ),
        # Coverage is ~100% below 12 sites, so a selection set skewed small
        # would not separate the settings the way the SI benchmarks do.
        (
            "tierA_nle20",
            ROOT / "data_prep/outputs/crystal_language_unified_tierA_v2/test.jsonl",
            args.tiera_limit,
            8,
        ),
    ]
    for name, src, limit, min_sites in jobs:
        if not src.exists():
            print(f"SKIP {name}: {src} missing")
            continue
        dst = OUT_ROOT / name / "crystal_language_v2/test.jsonl"
        n = carve(src, dst, args.max_sites, limit, min_sites=min_sites)
        print(f"{name}: {n} records -> {dst}")


if __name__ == "__main__":
    main()
