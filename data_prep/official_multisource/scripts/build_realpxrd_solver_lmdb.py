#!/usr/bin/env python3
"""Build gzipped LMDBs in RealPXRD-Solver's CrystDataset_pxrd format.

legacy: peak picking copies scripts/sample_flow_L.py; CIF cell stored as-is.
si:     paper SI preprocessor — Cu Kα via d-spacing, local-maxima peaks,
        primitive cell (see rps_si_preprocess.py).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
from numpy.polynomial import Polynomial
from pymatgen.core import Lattice, Structure
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def subtract_polynomial_background(continue_x, continue_y, poly_order, smoothing_window=3):
    smoothed_data = np.convolve(continue_y, np.ones(smoothing_window) / smoothing_window, mode="same")
    p = Polynomial.fit(continue_x, smoothed_data, poly_order)
    background = p(continue_x)
    intensity_corrected = smoothed_data - background
    intensity_corrected = np.maximum(0, intensity_corrected)
    vmax = float(np.max(intensity_corrected)) if intensity_corrected.size else 0.0
    if vmax <= 0:
        return intensity_corrected
    return intensity_corrected / vmax * 100


def search_peaks(continue_x, continue_y, intensity_threshold=3, min_peak_width=3, prominence=0.05):
    prominence_abs = (np.max(continue_y) - np.min(continue_y)) * prominence
    peaks_indices, _ = find_peaks(
        continue_y, height=intensity_threshold, width=min_peak_width, prominence=prominence_abs
    )
    pxrd_x = continue_x[peaks_indices]
    pxrd_y = continue_y[peaks_indices]
    if pxrd_y.size == 0:
        return pxrd_x, pxrd_y
    pxrd_y = pxrd_y / np.max(pxrd_y) * 100
    return pxrd_x, pxrd_y


def load_xy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open() as f:
        sample = f.read(256)
        f.seek(0)
        if "two_theta" in sample.lower() or "intensity" in sample.lower():
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    rows.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
        else:
            for line in f:
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"empty pattern: {path}")
    return arr[:, 0], arr[:, 1]


def rescale_peak_list(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return x, y
    ymax = float(np.max(y))
    if ymax <= 0:
        return x, y
    return x, y / ymax * 100.0


def peaks_from_profile(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return x, y
    y_corr = subtract_polynomial_background(x, y, poly_order=6)
    return search_peaks(x, y_corr)


def structure_payload(struct: Structure) -> dict:
    symbols = []
    for site in struct:
        sp = site.species.elements[0].symbol
        symbols.append(sp)
    return {
        "p_atom_type": symbols,
        "p_atom_pos": np.asarray(struct.frac_coords, dtype=np.float64),
        "p_lattice_matrix": np.asarray(struct.lattice.matrix, dtype=np.float64),
    }


def write_lmdb(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    env = lmdb.open(
        str(out_path),
        map_size=1 << 30,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    with env.begin(write=True) as txn:
        for rec in records:
            key = rec.pop("_key").encode("utf-8")
            payload = {k: v for k, v in rec.items() if k != "meta"}
            txn.put(key, gzip.compress(pickle.dumps(payload, protocol=4)))
    env.sync()
    env.close()


def convert_raw_dir(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    records = []
    meta = []
    cifs = sorted(raw_dir.glob("*.cif"))
    for cif in cifs:
        csv_path = cif.with_suffix(".csv")
        if not csv_path.exists():
            meta.append({"id": cif.stem, "status": "missing_csv"})
            continue
        try:
            struct = Structure.from_file(str(cif))
            x, y = load_xy_csv(csv_path)
            px, py = peaks_from_profile(x, y)
            if len(px) == 0:
                meta.append({"id": cif.stem, "status": "no_peaks", "n_sites": len(struct)})
                continue
            # BERT position embedding is floor(2θ); drop out-of-range.
            keep = (px >= 0) & (px < 180)
            px, py = px[keep], py[keep]
            if len(px) > 179:
                order = np.sort(np.argsort(py)[::-1][:179])
                px, py = px[order], py[order]
            rec = structure_payload(struct)
            rec["pxrd_x"] = np.asarray(px, dtype=np.float64)
            rec["pxrd_y"] = np.asarray(py, dtype=np.float64)
            rec["_key"] = cif.stem
            records.append(rec)
            meta.append(
                {
                    "id": cif.stem,
                    "status": "ok",
                    "formula": struct.composition.reduced_formula,
                    "n_sites": len(struct),
                    "n_peaks": int(len(px)),
                    "two_theta_min": float(px.min()) if len(px) else None,
                    "two_theta_max": float(px.max()) if len(px) else None,
                    "n_profile": int(len(x)),
                    "cell": "as_cif",
                }
            )
        except Exception as exc:  # noqa: BLE001
            meta.append({"id": cif.stem, "status": f"error:{type(exc).__name__}", "msg": str(exc)[:200]})
    return records, meta


def convert_raw_dir_si(raw_dir: Path, opxrd_dir: Path | None) -> tuple[list[dict], list[dict]]:
    from rps_si_preprocess import infer_and_convert, opxrd_hint, to_primitive

    records = []
    meta = []
    cifs = sorted(raw_dir.glob("*.cif"))
    for cif in cifs:
        csv_path = cif.with_suffix(".csv")
        if not csv_path.exists():
            meta.append({"id": cif.stem, "status": "missing_csv"})
            continue
        try:
            struct = to_primitive(Structure.from_file(str(cif)))
            x, y = load_xy_csv(csv_path)
            hint = opxrd_hint(cif, opxrd_dir)
            converted = infer_and_convert(x, y, struct, hinted_wavelength=hint)
            px = np.asarray(converted["pxrd_x"], dtype=np.float64)
            py = np.asarray(converted["pxrd_y"], dtype=np.float64)
            if len(px) == 0:
                meta.append(
                    {
                        "id": cif.stem,
                        "status": "no_peaks",
                        "n_sites": len(struct),
                        "axis": converted.get("kind"),
                        "wavelength": converted.get("wavelength"),
                    }
                )
                continue
            rec = structure_payload(struct)
            rec["pxrd_x"] = px
            rec["pxrd_y"] = py
            rec["_key"] = cif.stem
            records.append(rec)
            meta.append(
                {
                    "id": cif.stem,
                    "status": "ok",
                    "formula": struct.composition.reduced_formula,
                    "n_sites": len(struct),
                    "n_peaks": int(len(px)),
                    "two_theta_min": float(px.min()),
                    "two_theta_max": float(px.max()),
                    "n_profile": int(len(x)),
                    "n_profile_cu": converted.get("n_profile_cu"),
                    "axis": converted["kind"],
                    "wavelength": converted.get("wavelength"),
                    "axis_score": converted.get("score"),
                    "wavelength_hint": hint,
                    "cell": "primitive",
                }
            )
        except Exception as exc:  # noqa: BLE001
            meta.append({"id": cif.stem, "status": f"error:{type(exc).__name__}", "msg": str(exc)[:200]})
    return records, meta


def convert_native_jsonl(jsonl: Path) -> tuple[list[dict], list[dict]]:
    records = []
    meta = []
    for line in jsonl.open():
        row = json.loads(line)
        rid = row["id"]
        try:
            lattice = Lattice(row["lattice_matrix"])
            struct = Structure(lattice, row["atom_types"], row["frac_coords"])
            prim = struct.get_primitive_structure()
            if len(prim) <= len(struct):
                struct = prim
            exp = row.get("experimental") or {}
            raw_path = exp.get("opxrd_path") or exp.get("title")
            cond = row.get("conditions") or {}
            if raw_path and Path(raw_path).exists() and Path(raw_path).suffix == ".json":
                raw = json.loads(Path(raw_path).read_text())
                x = np.asarray(raw["two_theta_values"], dtype=np.float64)
                y = np.asarray(raw["intensities"], dtype=np.float64)
                px, py = peaks_from_profile(x, y)
            else:
                x = np.asarray(cond["pxrd_x"], dtype=np.float64)
                y = np.asarray(cond["pxrd_y"], dtype=np.float64)
                px, py = rescale_peak_list(x, y)
            keep = (px >= 0) & (px < 180)
            px, py = px[keep], py[keep]
            if len(px) > 179:
                order = np.sort(np.argsort(py)[::-1][:179])
                px, py = px[order], py[order]
            rec = structure_payload(struct)
            rec["pxrd_x"] = np.asarray(px, dtype=np.float64)
            rec["pxrd_y"] = np.asarray(py, dtype=np.float64)
            rec["_key"] = rid
            records.append(rec)
            meta.append(
                {
                    "id": rid,
                    "status": "ok",
                    "formula": struct.composition.reduced_formula,
                    "n_sites": len(struct),
                    "n_peaks": int(len(px)),
                    "from_raw_profile": bool(raw_path and Path(str(raw_path)).exists()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            meta.append({"id": rid, "status": f"error:{type(exc).__name__}", "msg": str(exc)[:200]})
    return records, meta


def dump_meta(meta: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in meta:
            f.write(json.dumps(row) + "\n")
    ok = [m for m in meta if m.get("status") == "ok"]
    print(f"wrote {path}  ok={len(ok)}/{len(meta)}", file=sys.stderr)
    if ok:
        ns = [m["n_sites"] for m in ok]
        npk = [m["n_peaks"] for m in ok]
        print(
            f"  n_sites median={np.median(ns):.0f} max={max(ns)}  "
            f"n_peaks median={np.median(npk):.0f} max={max(npk)}  n<=20={sum(n<=20 for n in ns)}",
            file=sys.stderr,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("legacy", "si"), default="legacy")
    p.add_argument("--out-dir", default="")
    p.add_argument("--sets", nargs="*", default=None)
    args = p.parse_args()
    if args.out_dir:
        out = Path(args.out_dir)
    else:
        out = ROOT / (
            "data_prep/realpxrd_solver/lmdb_si"
            if args.protocol == "si"
            else "data_prep/realpxrd_solver/lmdb"
        )
    opxrd_dir = ROOT / "data_prep/outputs/public_exp_pxrd/raw/opxrd/CNRS"

    jobs = [
        ("cnrs126", ROOT / "data_prep/realpxrd_solver/data/CNRS_raw_data", "raw"),
        ("rruff269", ROOT / "data_prep/realpxrd_solver/data/RRUFF_raw_data", "raw"),
        ("cnrs205", ROOT / "data_prep/outputs/cnrs_bench205/native/test.jsonl", "jsonl"),
        ("rruff217", ROOT / "data_prep/outputs/rruff_exp_pxrd/native/test.jsonl", "jsonl"),
    ]
    wanted = set(args.sets) if args.sets else None
    for name, src, kind in jobs:
        if wanted is not None and name not in wanted:
            continue
        if args.protocol == "si" and kind != "raw":
            print(f"SKIP {name}: SI protocol is defined on the released CIF+CSV", file=sys.stderr)
            continue
        if not src.exists():
            print(f"SKIP {name}: {src} missing", file=sys.stderr)
            continue
        print(f"=== {name} protocol={args.protocol} from {src} ===", file=sys.stderr)
        if kind == "raw" and args.protocol == "si":
            recs, meta = convert_raw_dir_si(src, opxrd_dir if "cnrs" in name else None)
        elif kind == "raw":
            recs, meta = convert_raw_dir(src)
        else:
            recs, meta = convert_native_jsonl(src)
        dump_meta(meta, out / f"{name}.meta.jsonl")
        if recs:
            write_lmdb(recs, out / f"{name}.lmdb")
            print(f"  lmdb {out / f'{name}.lmdb'} n={len(recs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
