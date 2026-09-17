#!/usr/bin/env python
"""Merge serial and sharded OQMD JSONL downloads into one deduplicated file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def key_for_row(row: dict[str, Any]) -> str:
    return ":".join(
        str(row.get(k, ""))
        for k in ("entry_id", "calculation_id", "formationenergy_id")
    )


def iter_rows(path: Path):
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_prep/raw/official_extra_sources")
    parser.add_argument("--out", default=None)
    parser.add_argument("--include-main", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out = Path(args.out or raw_dir / "oqmd_formationenergy_merged.jsonl")
    inputs: list[Path] = []
    main = raw_dir / "oqmd_formationenergy.jsonl"
    if args.include_main and main.exists():
        inputs.append(main)
    inputs.extend(sorted(raw_dir.glob("oqmd_formationenergy_shard_*_of_*.jsonl")))
    if not inputs:
        raise SystemExit(f"No OQMD shard files found in {raw_dir}")

    seen: set[str] = set()
    written = 0
    with out.open("w") as dst:
        for path in inputs:
            print(f"merge {path}")
            for row in tqdm(iter_rows(path), desc=path.name):
                k = key_for_row(row)
                if k in seen:
                    continue
                seen.add(k)
                dst.write(json.dumps(row) + "\n")
                written += 1
    print(f"merged rows: {written}")
    print(f"output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
