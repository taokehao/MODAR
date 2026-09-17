#!/usr/bin/env python
"""Merge per-source LMDB outputs into a single unified multimodal LMDB.

Inputs: one or more directories produced by ``02_build_lmdb_*.py`` (each
containing ``train.lmdb`` / ``valid.lmdb`` / ``test.lmdb``).

Output: a single directory with ``train.lmdb`` / ``valid.lmdb`` / ``test.lmdb``
that the model loader can read.  We rely on the build-time deterministic
``split_from_key`` so re-grouping per-source splits keeps the train/valid/test
membership stable even when sources are added or removed later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import LmdbWriter, dumps_record, iter_lmdb  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "data_prep" / "outputs" / "official_unified_multimodal"


def gather_source_dirs(arg_values: list[str]) -> list[Path]:
    paths = [Path(p) for p in arg_values]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing source dirs: {missing}")
    return paths


def iter_split(source_dirs: Iterable[Path], split: str):
    for src in source_dirs:
        path = src / f"{split}.lmdb"
        if not path.exists():
            continue
        for key, record in iter_lmdb(str(path), compressed=True):
            yield src.name, key, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="path to a per-source dataset dir (with train/valid/test.lmdb); "
        "repeat for each source.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument("--dedupe", action="store_true",
                        help="dedupe across sources via record['structure_hash']")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = gather_source_dirs(args.source)
    print(f"Merging {len(source_dirs)} source datasets -> {out_dir}")
    for p in source_dirs:
        print(f"  {p}")

    counts: Counter = Counter()
    seen_hashes: set[str] = set()
    writers = {
        split: LmdbWriter(
            str(out_dir / f"{split}.lmdb"),
            map_size=args.map_size_gb * 1024**3,
            commit_every=1000,
            compress=True,
        )
        for split in ("train", "valid", "test")
    }
    try:
        for split, writer in writers.items():
            for src_name, key, record in tqdm(
                iter_split(source_dirs, split), desc=f"merge {split}"
            ):
                if args.dedupe:
                    h = record.get("structure_hash")
                    if h is not None:
                        if h in seen_hashes:
                            counts[f"{src_name}:{split}:deduped"] += 1
                            continue
                        seen_hashes.add(h)
                # Re-key with `${source}:${source_id}` to avoid cross-source key collisions.
                source = record.get("source", src_name)
                source_id = record.get("source_id") or key.decode("utf-8", "ignore")
                new_key = f"{source}:{source_id}".encode("utf-8")
                writer.put(record, key=new_key)
                counts[f"{src_name}:{split}"] += 1
                counts[f"total:{split}"] += 1
    finally:
        for writer in writers.values():
            writer.close()

    report = {
        "counts": dict(counts),
        "sources": [str(p) for p in source_dirs],
        "args": vars(args),
    }
    (out_dir / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Unified LMDB -> {out_dir}")
    return 0


# Ensure ``LmdbWriter`` and ``dumps_record`` stay importable for downstream code.
__all__ = ["LmdbWriter", "dumps_record"]


if __name__ == "__main__":
    raise SystemExit(main())
