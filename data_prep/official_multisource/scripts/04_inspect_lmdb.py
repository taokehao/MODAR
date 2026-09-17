#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lmdb_utils import iter_lmdb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lmdb")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)

    for i, (key, rec) in enumerate(iter_lmdb(args.lmdb)):
        if i >= args.n:
            break
        print(f"--- record {i} key={key!r} ---")
        for k, v in rec.items():
            if hasattr(v, "shape"):
                print(f"{k:30s}: {type(v).__name__} shape={v.shape} dtype={v.dtype}")
            else:
                text = repr(v)
                if len(text) > 120:
                    text = text[:117] + "..."
                print(f"{k:30s}: {type(v).__name__} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
