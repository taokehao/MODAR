#!/usr/bin/env python
"""Add simulated neutron diffraction and XPS-style valence fields to raw JSONL.

This is an offline enrichment step between raw download and LMDB construction.
It keeps training fast and makes repeated LMDB builds cheaper because neutron
patterns no longer need to be simulated each time.

The script never edits the input file in-place by default.  It writes a new
``*.multimodal.jsonl`` file with the original JSON row plus:

  neutron_x, neutron_y
  valence_assignments, valence_source, valence_assigned

Supported source parsers:
  alexandria  - OPTIMADE rows from Alexandria
  oqmd        - rows from OQMD formationenergy endpoint
  aflow       - AFLOW per-entry rows downloaded by 01_download_extra_sources.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.multimodal_utils import compute_multimodal_signals  # noqa: E402

DEFAULT_RAW = PROJECT_ROOT / "data_prep" / "raw" / "official_extra_sources"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extra_build = _load_module(
    ROOT / "scripts" / "02_build_lmdb_extra_sources.py",
    "_official_build_extra_sources",
)
_aflow_build = _load_module(
    ROOT / "scripts" / "02_build_lmdb_aflow.py",
    "_official_build_aflow",
)


def parser_for_source(source: str) -> Callable[[dict[str, Any]], Any | None]:
    if source == "alexandria":
        return _extra_build.alexandria_structure_from_optimade
    if source == "oqmd":
        return _extra_build.oqmd_structure_from_row
    if source == "aflow":
        return _aflow_build.aflow_structure_from_row
    raise ValueError(f"Unsupported source: {source}")


def default_output_path(input_jsonl: Path) -> Path:
    suffix = "".join(input_jsonl.suffixes)
    if suffix.endswith(".jsonl"):
        return input_jsonl.with_name(input_jsonl.name[: -len(".jsonl")] + ".multimodal.jsonl")
    return input_jsonl.with_suffix(input_jsonl.suffix + ".multimodal.jsonl")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def enrich_jsonl(
    *,
    source: str,
    input_jsonl: Path,
    output_jsonl: Path,
    max_records: int | None = None,
    resume: bool = True,
    two_theta_min: float = 5.0,
    two_theta_max: float = 80.0,
) -> dict[str, int]:
    parse_structure = parser_for_source(source)
    done = count_lines(output_jsonl) if resume else 0
    mode = "a" if done else "w"
    counts = {
        "seen": 0,
        "written": done,
        "skipped_existing": done,
        "parse_failed": 0,
        "neutron_missing": 0,
        "valence_missing": 0,
    }

    with input_jsonl.open() as src, output_jsonl.open(mode) as dst:
        for line_i, line in enumerate(tqdm(src, desc=f"enrich {source}")):
            if line_i < done:
                continue
            if max_records is not None and counts["written"] - done >= max_records:
                break
            counts["seen"] += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["parse_failed"] += 1
                continue
            try:
                structure = parse_structure(row)
            except Exception:
                structure = None
            if structure is None:
                counts["parse_failed"] += 1
                continue

            mm = compute_multimodal_signals(
                structure,
                two_theta_min=two_theta_min,
                two_theta_max=two_theta_max,
                enable_neutron=True,
                enable_valence=True,
            )
            if mm.get("neutron_x") is None or mm.get("neutron_y") is None:
                counts["neutron_missing"] += 1
            if not mm.get("valence_assigned"):
                counts["valence_missing"] += 1

            # JSON cannot serialize numpy arrays.
            if hasattr(mm.get("neutron_x"), "tolist"):
                mm["neutron_x"] = mm["neutron_x"].tolist()
            if hasattr(mm.get("neutron_y"), "tolist"):
                mm["neutron_y"] = mm["neutron_y"].tolist()

            row.update(mm)
            dst.write(json.dumps(row) + "\n")
            counts["written"] += 1
            if counts["written"] % 1000 == 0:
                dst.flush()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["alexandria", "oqmd", "aflow"])
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--two-theta-min", type=float, default=5.0)
    parser.add_argument("--two-theta-max", type=float, default=80.0)
    args = parser.parse_args(argv)

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_output_path(input_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not input_jsonl.exists():
        raise SystemExit(f"Input JSONL not found: {input_jsonl}")

    print(f"source: {args.source}")
    print(f"input:  {input_jsonl}")
    print(f"output: {output_jsonl}")
    counts = enrich_jsonl(
        source=args.source,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        max_records=args.max_records,
        resume=not args.no_resume,
        two_theta_min=args.two_theta_min,
        two_theta_max=args.two_theta_max,
    )
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
