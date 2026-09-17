#!/usr/bin/env python
"""Unified entrypoint for the prepared multimodal crystal dataset.

The download/build/enrichment scripts are preserved for provenance, but the
dataset is already prepared. Day-to-day work should use this script to inspect
the final unified LMDB or export it to crystal-native JSONL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_UNIFIED_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "official_unified_multimodal"
DEFAULT_JSONL_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "crystal_native_unified"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_status(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    merge_report = read_json(data_dir / "merge_report.json")
    stats = read_json(data_dir / "normalization_stats.json")
    counts = merge_report.get("counts", {})
    source_counts = stats.get("source_counts", {})

    summary = {
        "data_dir": str(data_dir),
        "splits": {
            "train": counts.get("total:train"),
            "valid": counts.get("total:valid"),
            "test": counts.get("total:test"),
        },
        "train_source_counts": source_counts,
        "files_present": {
            split: (data_dir / f"{split}.lmdb").exists()
            for split in ("train", "valid", "test")
        },
        "normalization_stats": str(data_dir / "normalization_stats.json"),
        "merge_report": str(data_dir / "merge_report.json"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    lmdb_path = Path(args.data_dir) / f"{args.split}.lmdb"
    if not lmdb_path.exists():
        raise FileNotFoundError(f"missing LMDB split: {lmdb_path}")
    script = ROOT / "scripts" / "04_inspect_lmdb.py"
    return subprocess.call(
        [sys.executable, str(script), str(lmdb_path), "--n", str(args.n)]
    )


def cmd_export_jsonl(args: argparse.Namespace) -> int:
    script = ROOT / "scripts" / "export_crystal_native_jsonl.py"
    command = [
        sys.executable,
        str(script),
        "--data-dir",
        str(Path(args.data_dir)),
        "--out-dir",
        str(Path(args.out_dir)),
    ]
    if args.max_records_per_split is not None:
        command.extend(["--max-records-per-split", str(args.max_records_per_split)])
    return subprocess.call(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="summarize final unified data")
    status.add_argument("--data-dir", default=str(DEFAULT_UNIFIED_DIR))
    status.set_defaults(func=cmd_status)

    inspect = subparsers.add_parser("inspect", help="print a few LMDB records")
    inspect.add_argument("--data-dir", default=str(DEFAULT_UNIFIED_DIR))
    inspect.add_argument("--split", choices=["train", "valid", "test"], default="train")
    inspect.add_argument("--n", type=int, default=3)
    inspect.set_defaults(func=cmd_inspect)

    export_jsonl = subparsers.add_parser(
        "export-jsonl", help="export unified LMDB to crystal-native JSONL"
    )
    export_jsonl.add_argument("--data-dir", default=str(DEFAULT_UNIFIED_DIR))
    export_jsonl.add_argument("--out-dir", default=str(DEFAULT_JSONL_DIR))
    export_jsonl.add_argument("--max-records-per-split", type=int, default=None)
    export_jsonl.set_defaults(func=cmd_export_jsonl)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
