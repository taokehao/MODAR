#!/usr/bin/env python3
"""Convert the RealPXRD-Solver CNRS/RRUFF releases into crystal_language_v2 eval sets.

Their release ships one CIF plus one dense CSV pattern per target. This turns each
pair into the same native + sequence records our own benchmarks use, so the two
models can be scored on identical targets.

Two PXRD front-ends are emitted because they are not interchangeable:
  profile  rolling-percentile background + 0.05 deg binning (what cnrs_bench205 and
           the public 337 use, so numbers stay comparable with our existing runs)
  peaks    SI local-maxima (protocol=si) or sample_flow_L.py (protocol=legacy)

legacy: CIF cell stored as-is, no wavelength conversion.
si:     primitive cell + Cu Kα remap (rps_si_preprocess.py). Output dirs get an
        `_si` infix so the legacy sets are not overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "data_prep/official_multisource/scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep/official_multisource"))
sys.path.insert(0, str(SCRIPTS))

import build_collaborator_exp_pxrd as collab  # noqa: E402
import build_public_exp_pxrd as pub  # noqa: E402
from build_realpxrd_solver_lmdb import load_xy_csv, peaks_from_profile  # noqa: E402
from rps_si_preprocess import CU_KA, infer_and_convert, opxrd_hint, to_primitive  # noqa: E402

from pymatgen.core import Structure  # noqa: E402


def encode_profile_front_end(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float], int]:
    x_clip, _, peak_x, peak_y = collab.encode_profile(x, y)
    return list(peak_x), list(peak_y), len(x_clip)


def encode_peaks_front_end(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float], int]:
    px, py = peaks_from_profile(x, y)
    keep = (px >= pub.GRID_MIN) & (px <= pub.GRID_MAX)
    px, py = px[keep], py[keep]
    if py.size:
        py = py / float(np.max(py))
    return [float(v) for v in px], [float(v) for v in py], int(len(x))


ENCODERS = {"profile": encode_profile_front_end, "peaks": encode_peaks_front_end}


def _peaks_from_si(converted: dict) -> tuple[list[float], list[float], int]:
    px = np.asarray(converted["pxrd_x"], dtype=np.float64)
    py = np.asarray(converted["pxrd_y"], dtype=np.float64)
    if py.size:
        py = py / float(np.max(py))
    return [float(v) for v in px], [float(v) for v in py], int(converted.get("n_profile_cu") or len(px))


def build(
    raw_dir: Path,
    prefix: str,
    encoding: str,
    min_peaks: int,
    protocol: str,
    opxrd_dir: Path | None,
) -> tuple[list[dict], list[dict]]:
    encode = ENCODERS[encoding]
    records: list[dict] = []
    meta: list[dict] = []
    for cif in sorted(raw_dir.glob("*.cif")):
        csv_path = cif.with_suffix(".csv")
        rid = f"{prefix}:{cif.stem}"
        if not csv_path.exists():
            meta.append({"id": rid, "status": "missing_csv"})
            continue
        try:
            structure = Structure.from_file(str(cif))
            x, y = load_xy_csv(csv_path)
            axis_kind = None
            axis_wavelength = pub.REF_WAVELENGTH
            axis_score = None
            if protocol == "si":
                structure = to_primitive(structure)
                hint = opxrd_hint(cif, opxrd_dir) if prefix == "rps_cnrs" else None
                converted = infer_and_convert(x, y, structure, hinted_wavelength=hint)
                axis_kind = converted.get("kind")
                axis_wavelength = float(converted.get("wavelength") or CU_KA)
                axis_score = converted.get("score")
                if encoding == "peaks":
                    peak_x, peak_y, n_profile = _peaks_from_si(converted)
                else:
                    x_cu = np.asarray(converted["x_cu"], dtype=np.float64)
                    y_cu = np.asarray(converted["y_cu"], dtype=np.float64)
                    peak_x, peak_y, n_profile = encode_profile_front_end(x_cu, y_cu)
                wavelength = CU_KA
                kind = f"experimental_si_{encoding}"
            else:
                peak_x, peak_y, n_profile = encode(x, y)
                wavelength = pub.REF_WAVELENGTH
                kind = f"experimental_{encoding}"
            if len(peak_x) < min_peaks:
                meta.append({"id": rid, "status": "too_few_peaks", "n_peaks": len(peak_x)})
                continue
            rec = pub.to_native_record(
                sample_id=rid,
                source="realpxrd_solver_release",
                source_id=cif.stem,
                row={},
                structure=structure,
                peak_x=peak_x,
                peak_y=peak_y,
                pattern_meta={"wavelength": wavelength, "kind": kind},
                n_profile=n_profile,
                q_aligned=False,
                disordered=False,
            )
            records.append(rec)
            row = {
                "id": rid,
                "status": "ok",
                "formula": rec["formula"],
                "n_sites": rec["n_sites"],
                "space_group": rec["space_group"],
                "n_peaks": len(peak_x),
                "simulated_pxrd_cosine": rec["experimental"]["simulated_pxrd_cosine"],
                "protocol": protocol,
            }
            if protocol == "si":
                row.update(
                    {
                        "axis": axis_kind,
                        "axis_wavelength": axis_wavelength,
                        "axis_score": axis_score,
                        "cell": "primitive",
                    }
                )
            meta.append(row)
        except Exception as exc:  # noqa: BLE001
            meta.append({"id": rid, "status": f"error:{type(exc).__name__}", "msg": str(exc)[:200]})
    return records, meta


def report(name: str, meta: list[dict], n_seq: int, n_failed: int) -> dict:
    ok = [m for m in meta if m.get("status") == "ok"]
    ns = [m["n_sites"] for m in ok]
    npk = [m["n_peaks"] for m in ok]
    cos = [m["simulated_pxrd_cosine"] for m in ok if m.get("simulated_pxrd_cosine") is not None]
    summary = {
        "name": name,
        "n_input": len(meta),
        "n_ok": len(ok),
        "n_sequences": n_seq,
        "n_sequence_failed": n_failed,
        "n_sites_median": float(np.median(ns)) if ns else None,
        "n_sites_max": int(max(ns)) if ns else None,
        "n_sites_le20": int(sum(n <= 20 for n in ns)),
        "n_peaks_median": float(np.median(npk)) if npk else None,
        "pxrd_cosine_median": float(np.median(cos)) if cos else None,
    }
    print(json.dumps(summary), file=sys.stderr)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(ROOT / "data_prep/outputs"))
    p.add_argument("--protocol", choices=("legacy", "si"), default="legacy")
    p.add_argument("--encodings", nargs="+", default=["profile", "peaks"], choices=sorted(ENCODERS))
    p.add_argument("--sets", nargs="*", default=None, help="cnrs126 and/or rruff269; default both")
    # Their eval keeps every target, so default to no peak-count gate; raising this
    # reproduces the stricter filter our own benchmarks use.
    p.add_argument("--min-peaks", type=int, default=1)
    args = p.parse_args()

    jobs = [
        ("cnrs126", ROOT / "data_prep/realpxrd_solver/data/CNRS_raw_data", "rps_cnrs"),
        ("rruff269", ROOT / "data_prep/realpxrd_solver/data/RRUFF_raw_data", "rps_rruff"),
    ]
    out_root = Path(args.out_root)
    opxrd_dir = ROOT / "data_prep/outputs/public_exp_pxrd/raw/opxrd/CNRS"
    wanted = set(args.sets) if args.sets else None
    summaries = []
    for name, raw_dir, prefix in jobs:
        if wanted is not None and name not in wanted:
            continue
        if not raw_dir.exists():
            print(f"SKIP {name}: {raw_dir} missing", file=sys.stderr)
            continue
        for encoding in args.encodings:
            tag = f"rps_{name}_{'si_' if args.protocol == 'si' else ''}{encoding}"
            print(f"=== {tag} protocol={args.protocol} ===", file=sys.stderr)
            records, meta = build(
                raw_dir,
                prefix,
                encoding,
                args.min_peaks,
                args.protocol,
                opxrd_dir if prefix == "rps_cnrs" else None,
            )
            out_dir = out_root / tag
            pub.write_jsonl(out_dir / "native/test.jsonl", records)
            pub.write_jsonl(out_dir / "meta.jsonl", meta)
            n_seq, n_failed = pub.write_sequences(records, out_dir / "crystal_language_v2/test.jsonl")
            s = report(tag, meta, n_seq, n_failed)
            s["out_dir"] = str(out_dir)
            s["encoding"] = encoding
            s["protocol"] = args.protocol
            summaries.append(s)
            (out_dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    summary_name = "rps_si_bench_summary.json" if args.protocol == "si" else "rps_bench_summary.json"
    (out_root / summary_name).write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
