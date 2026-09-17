#!/usr/bin/env python
"""Locate or download MP/WBM raw files for the official_multisource_props dataset.

WBM and the MP energy tables are public via Matbench Discovery/Figshare. Some
clusters receive a Figshare/AWS WAF challenge and cache zero-byte files. For
that case, download the files manually on a machine with browser access and
place them in --raw-dir, or pass explicit --wbm-* paths.

MP structures are preferably pulled through the official Materials Project API
because the API gives the latest structure + property docs in one pass.

Usage:
  python 01_download_mp_wbm.py --locate-wbm
  MP_API_KEY=... python 01_download_mp_wbm.py --download-mp --mp-limit 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import sleep

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "raw" / "official_mp_wbm"


def ensure_matbench_discovery():
    try:
        from matbench_discovery.data import DataFiles  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "matbench-discovery is required for WBM public data. Install with:\n"
            "  conda activate uni3dar\n"
            "  pip install matbench-discovery\n"
            f"Original error: {exc!r}"
        )


def _nonempty(path: Path | str | None) -> bool:
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 0


def _find_first(raw_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(raw_dir.rglob(name))
        for path in matches:
            if path.stat().st_size > 0:
                return path
    return None


def locate_wbm(raw_dir: Path, try_matbench: bool = False) -> dict:
    """Return paths to WBM files, trying raw_dir first and matbench cache second."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary = _find_first(
        raw_dir,
        [
            "*wbm*summary*.csv.gz",
            "2022-10-19-wbm-summary.csv.gz",
            "2023-12-13-wbm-summary.csv.gz",
            "wbm-summary.csv.gz",
            "wbm_summary.csv.gz",
        ],
    )
    structs = _find_first(
        raw_dir,
        [
            "*wbm*init*struct*.jsonl.gz",
            "*wbm*init*struct*.json.bz2",
            "*wbm*init*struct*.jsonl.bz2",
            "2022-10-19-wbm-init-structs.json.bz2",
            "2022-10-19-wbm-init-structs.jsonl.gz",
            "wbm-init-structs.json.bz2",
            "wbm_initial_structures.jsonl",
            "wbm_initial_structures.jsonl.gz",
        ],
    )
    entries = _find_first(
        raw_dir,
        [
            "*wbm*computed*structure*entries*.jsonl.gz",
            "*wbm*computed*structure*entries*.json.bz2",
            "2022-10-19-wbm-computed-structure-entries.json.bz2",
            "wbm-computed-structure-entries.json.bz2",
        ],
    )
    mp_energies = _find_first(raw_dir, ["2025-02-01-mp-energies.csv.gz", "2023-01-10-mp-energies.csv.gz"])

    if try_matbench and (summary is None or structs is None):
        located = download_wbm_with_matbench(raw_dir)
        summary = summary or Path(located.get("wbm_summary", ""))
        structs = structs or Path(located.get("wbm_initial_structures", ""))
        entries = entries or Path(located.get("wbm_computed_structure_entries", ""))
        mp_energies = mp_energies or Path(located.get("mp_energies", ""))

    missing = []
    if not _nonempty(summary):
        missing.append("WBM summary CSV")
    if not _nonempty(structs) and not _nonempty(entries):
        missing.append("WBM initial structures JSONL/BZ2 or computed-structure-entries JSON BZ2")
    if missing:
        raise SystemExit(
            "Missing required WBM files in --raw-dir. Needed: "
            + ", ".join(missing)
            + "\nSuggested manual downloads (browser if server is blocked by Figshare WAF):\n"
            + "  wbm summary: https://figshare.com/files/44225498 or matbench-discovery registry file 41296866\n"
            + "  wbm init structures: https://figshare.com/files/53161835 or registry file 40344466\n"
            + f"Place files under: {raw_dir}"
        )

    out = {
        "wbm_summary": str(summary),
        "wbm_initial_structures": str(structs) if _nonempty(structs) else None,
        "wbm_computed_structure_entries": str(entries) if _nonempty(entries) else None,
        "mp_energies": str(mp_energies) if _nonempty(mp_energies) else None,
    }
    for name, path in out.items():
        if path:
            p = Path(path)
            print(f"{name}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return out


def download_wbm_with_matbench(raw_dir: Path) -> dict:
    ensure_matbench_discovery()
    from matbench_discovery.data import DataFiles

    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "wbm_summary": DataFiles.wbm_summary.path,
        "wbm_initial_structures": DataFiles.wbm_initial_structures.path,
        "wbm_computed_structure_entries": DataFiles.wbm_computed_structure_entries.path,
        "mp_energies": DataFiles.mp_energies.path,
    }
    out = {}
    for name, path in files.items():
        path = Path(path)
        if path.stat().st_size == 0:
            raise SystemExit(
                f"Downloaded {path} is empty. Figshare likely returned an AWS WAF challenge. "
                "Delete the empty cache file and download manually with a browser."
            )
        out[name] = str(path)
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return out


def download_mp_via_api(raw_dir: Path, api_key: str, limit: int | None, chunk_size: int = 1000) -> Path:
    out = raw_dir / ("mp_summary_sample.jsonl" if limit else "mp_summary.jsonl")
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "nsites",
        "symmetry",
    ]
    print(f"Querying MP summary docs via REST -> {out}")
    print(f"Fields: {fields}")
    count = 0
    url = "https://api.materialsproject.org/materials/summary/"
    session = requests.Session()
    session.headers.update(
        {
            "x-api-key": api_key,
            "user-agent": "pxrd_mof-official-mp-wbm/0.1",
        }
    )
    skip = 0
    with out.open("w") as f:
        while True:
            current_limit = min(chunk_size, limit - count) if limit is not None else chunk_size
            if current_limit <= 0:
                break
            params = {
                "_fields": ",".join(fields),
                "_limit": current_limit,
                "_skip": skip,
            }
            for attempt in range(5):
                resp = session.get(url, params=params, timeout=60)
                if resp.status_code in (429, 502, 504):
                    wait_s = 5 * (attempt + 1)
                    print(f"  MP API {resp.status_code}, retrying in {wait_s}s")
                    sleep(wait_s)
                    continue
                resp.raise_for_status()
                break
            payload = resp.json()
            docs = payload.get("data", [])
            if not docs:
                break
            for doc in docs:
                if limit is not None and count >= limit:
                    break
                f.write(json.dumps(doc) + "\n")
                count += 1
            skip += len(docs)
            if count % 1000 == 0 or limit is not None:
                total = payload.get("meta", {}).get("total_doc")
                print(f"  wrote {count}; skip={skip}; total={total}")
            if len(docs) < current_limit:
                break
    print(f"MP docs written: {count}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW))
    parser.add_argument("--locate-wbm", action="store_true")
    parser.add_argument("--try-matbench-download", action="store_true")
    parser.add_argument("--download-mp", action="store_true")
    parser.add_argument("--mp-limit", type=int, default=None, help="debug limit for MP API docs")
    parser.add_argument("--mp-chunk-size", type=int, default=1000)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    manifest["raw_dir"] = str(raw_dir)

    if args.locate_wbm:
        manifest["wbm"] = locate_wbm(raw_dir, try_matbench=args.try_matbench_download)

    if args.download_mp:
        api_key = os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")
        if not api_key:
            raise SystemExit("Set MP_API_KEY (or MAPI_KEY) to download MP structures via the official API.")
        manifest["mp_summary_jsonl"] = str(
            download_mp_via_api(raw_dir, api_key, args.mp_limit, args.mp_chunk_size)
        )

    if not args.locate_wbm and not args.download_mp:
        parser.error("Choose at least one of --locate-wbm / --download-mp")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
