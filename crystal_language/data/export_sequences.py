#!/usr/bin/env python
"""Export canonical crystal-language sequences from crystal-native JSONL.

Input is the read-only multimodal bridge produced for the FlowMM/DiffCSP route
(``data_prep/outputs/crystal_native_mp_wbm``). Output is an independent set of
sequence-ready JSONL files for the language-model route. The source files are
never modified, so this stays parallel to FlowMM.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.data.sequence import build_sequence_record  # noqa: E402

PROJECT_ROOT = PACKAGE_ROOT
DEFAULT_IN_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "crystal_native_mp_wbm"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "crystal_language_mp_wbm"


def export_split(
    in_path: Path, out_path: Path, max_records: int | None, rep: str = "v1"
) -> dict[str, int]:
    stats: Counter[str] = Counter()
    with in_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for i, line in enumerate(src):
            if max_records is not None and stats["written"] >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                seq = build_sequence_record(record, rep=rep)
            except Exception as exc:
                stats["skipped"] += 1
                stats[f"skip:{type(exc).__name__}"] += 1
                continue
            dst.write(json.dumps(seq, ensure_ascii=False, separators=(",", ":")))
            dst.write("\n")
            stats["written"] += 1
            for name in seq["cond"]:
                stats[f"cond:{name}"] += 1
    return dict(stats)


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [s for s in splits if s not in {"train", "valid", "test"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid splits: {invalid}")
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--splits", type=parse_splits, default=parse_splits("train,valid,test"))
    parser.add_argument("--max-records-per-split", type=int, default=None)
    parser.add_argument("--rep", choices=["v1", "v2"], default="v1")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "max_records_per_split": args.max_records_per_split,
        "rep": args.rep,
        "splits": {},
    }
    for split in args.splits:
        in_path = in_dir / f"{split}.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(f"missing input split: {in_path}")
        out_path = out_dir / f"{split}.jsonl"
        summary["splits"][split] = export_split(
            in_path, out_path, args.max_records_per_split, rep=args.rep
        )

    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
