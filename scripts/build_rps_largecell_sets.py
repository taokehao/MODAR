#!/usr/bin/env python3
"""Carve n_sites>=13 SI subsets for adaptive-N resampling."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
JOBS = [
    (
        ROOT / "data_prep/outputs/rps_cnrs126_si_profile/crystal_language_v2/test.jsonl",
        ROOT / "data_prep/outputs/rps_adapt_n/cnrs_profile_n13/crystal_language_v2/test.jsonl",
    ),
    (
        ROOT / "data_prep/outputs/rps_rruff269_si_peaks/crystal_language_v2/test.jsonl",
        ROOT / "data_prep/outputs/rps_adapt_n/rruff_peaks_n13/crystal_language_v2/test.jsonl",
    ),
]


def carve(src: Path, dst: Path, min_sites: int = 13) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            row = json.loads(line)
            if int(row.get("n_sites") or 0) < min_sites:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def main() -> None:
    for src, dst in JOBS:
        n = carve(src, dst)
        print(f"{n:4d}  {dst}")


if __name__ == "__main__":
    main()
