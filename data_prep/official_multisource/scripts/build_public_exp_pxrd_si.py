#!/usr/bin/env python3
"""Re-encode the frozen public experimental PXRD set (337 targets) under the SI protocol.

Same 337 targets and ids as ``public_exp_pxrd`` (no re-selection), but the PXRD
condition is rebuilt the way the RealPXRD-Solver CNRS/RRUFF benchmark sets were
(``build_realpxrd_solver_bench_v2.py --protocol si``):

* axis / wavelength inferred by matching the paired structure, then remapped to
  Cu Kα (``rps_si_preprocess.infer_and_convert``) instead of trusting the
  declared wavelength;
* primitive cell as the target (the flow baseline and our benchmark runs use it);
* the same profile front-end (rolling-percentile background + 0.05° bins).

ND / XRF / XPS conditions are re-simulated from the (primitive) structure exactly
as in the original builder. Output: ``public_exp_pxrd_si/{native,crystal_language_v2}``
plus ``meta.jsonl`` with the inferred axis per target.
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

import build_public_exp_pxrd as pub  # noqa: E402
from rps_si_preprocess import CU_KA, infer_and_convert, to_primitive  # noqa: E402

from pymatgen.core import Lattice, Structure  # noqa: E402


def raw_pattern(rec: dict, raw_root: Path) -> tuple[np.ndarray, np.ndarray, float | None, str]:
    """Reload the raw observed pattern of a native record (opXRD json or COD cif/hkl)."""
    exp = rec["experimental"]
    if rec["source"].startswith("opxrd"):
        path = Path(exp.get("opxrd_path") or exp["title"])
        if not path.is_absolute():
            path = ROOT / path
        data = pub._walk_json(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        xs = pub._as_float_list(data.get("two_theta_values") or data.get("two_theta") or data.get("x") or data.get("angles"))
        ys = pub._as_float_list(data.get("intensities") or data.get("y") or data.get("intensity"))
        hint = exp.get("wavelength_A")
        return np.asarray(xs, float), np.asarray(ys, float), (float(hint) if hint else None), "profile"
    cod_id = str(rec["source_id"]).zfill(7)
    cif_path = raw_root / "cod" / f"{cod_id}.cif"
    hkl_path = raw_root / "cod" / f"{cod_id}.hkl"
    texts = [cif_path.read_text(encoding="utf-8", errors="replace")]
    if hkl_path.exists() and hkl_path.stat().st_size > 0:
        texts.append(hkl_path.read_text(encoding="utf-8", errors="replace"))
    pattern = pub.extract_observed_pxrd(texts, None)
    if pattern is None:
        raise ValueError("no observed pattern")
    return np.asarray(pattern["x"], float), np.asarray(pattern["y"], float), pattern.get("wavelength"), str(pattern["kind"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(ROOT / "data_prep/outputs/public_exp_pxrd"))
    ap.add_argument("--out", default=str(ROOT / "data_prep/outputs/public_exp_pxrd_si"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    (out / "native").mkdir(parents=True, exist_ok=True)
    (out / "crystal_language_v2").mkdir(parents=True, exist_ok=True)

    records, meta = [], []
    with (src / "native_scored_unique/test.jsonl").open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    for i, rec in enumerate(rows):
        rid = rec["id"]
        try:
            structure = Structure(Lattice(np.asarray(rec["lattice_matrix"], float)), rec["atom_types"], rec["frac_coords"])
            x, y, hint, kind = raw_pattern(rec, src / "raw")
            prim = to_primitive(structure)
            dense = "profile" in kind or x.size > 200
            if dense:
                conv = infer_and_convert(x, y, prim, hinted_wavelength=hint)
                x_cu = np.asarray(conv["x_cu"], float)
                y_cu = np.asarray(conv["y_cu"], float)
                x_clip, _, peak_x, peak_y = pub.collab.encode_profile(x_cu, y_cu)
                n_profile = int(len(x_clip))
                axis = {"axis": conv.get("kind"), "axis_wavelength": conv.get("wavelength"), "axis_score": conv.get("score")}
                pkind = "experimental_si_profile"
            else:
                # sparse peak lists (2 COD entries): keep the declared wavelength, remap to Cu Kα
                xs, ys, _ = pub.npd.remap_two_theta_to_train(x, y, float(hint or pub.REF_WAVELENGTH), CU_KA)
                x_clip, _, peak_x, peak_y = pub.npd.clip_and_peaks(xs, ys, dense_profile=False)
                n_profile = int(len(x_clip))
                axis = {"axis": "declared", "axis_wavelength": hint, "axis_score": None}
                pkind = "experimental_si_peaks"
            if len(peak_x) < pub.MIN_PEAKS:
                meta.append({"id": rid, "status": f"too_few_pxrd_bins:{len(peak_x)}"})
                continue
            exp = rec["experimental"]
            row = {
                "formula": exp.get("catalog_formula"), "doi": exp.get("doi"), "title": exp.get("title"),
                "year": str(exp.get("year") or ""), "journal": exp.get("journal"), "sgNumber": "",
            }
            native = pub.to_native_record(
                sample_id=rid, source=rec["source"], source_id=rec["source_id"], row=row, structure=prim,
                peak_x=list(peak_x), peak_y=list(peak_y), pattern_meta={"wavelength": CU_KA, "kind": pkind},
                n_profile=n_profile, q_aligned=True, disordered=bool(exp.get("disordered_cif")),
            )
            native["experimental"]["opxrd_path"] = exp.get("opxrd_path")
            native["experimental"]["conventional_n_sites"] = rec["n_sites"]
            records.append(native)
            meta.append({
                "id": rid, "status": "ok", "formula": native["formula"], "n_sites": native["n_sites"],
                "n_sites_conventional": rec["n_sites"], "space_group": native["space_group"],
                "n_peaks": len(peak_x), "simulated_pxrd_cosine": native["experimental"]["simulated_pxrd_cosine"],
                "cosine_before": exp.get("simulated_pxrd_cosine"), "protocol": "si", "cell": "primitive", **axis,
            })
        except Exception as exc:  # noqa: BLE001
            meta.append({"id": rid, "status": f"error:{type(exc).__name__}", "msg": str(exc)[:200]})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} ok={len(records)}", flush=True)

    pub.write_jsonl(out / "native/test.jsonl", records)
    n_seq, n_failed = pub.write_sequences(records, out / "crystal_language_v2/test.jsonl")
    with (out / "meta.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    ok = [m for m in meta if m.get("status") == "ok"]
    summary = {
        "source": str(src), "protocol": "si", "n_input": len(rows), "n_native": len(records),
        "n_sequences": n_seq, "n_sequence_failed": n_failed,
        "n_sites_median_primitive": float(np.median([m["n_sites"] for m in ok])) if ok else None,
        "n_sites_median_conventional": float(np.median([m["n_sites_conventional"] for m in ok])) if ok else None,
        "n_le20_primitive": int(sum(m["n_sites"] <= 20 for m in ok)),
        "cosine_median_before": float(np.median([m["cosine_before"] for m in ok if m.get("cosine_before") is not None])) if ok else None,
        "cosine_median_after": float(np.median([m["simulated_pxrd_cosine"] for m in ok if m.get("simulated_pxrd_cosine") is not None])) if ok else None,
        "axis_kinds": {k: sum(1 for m in ok if m.get("axis") == k) for k in sorted({m.get("axis") for m in ok})},
        "statuses": {k: sum(1 for m in meta if m.get("status") == k) for k in sorted({m.get("status") for m in meta})},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
