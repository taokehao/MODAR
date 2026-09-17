#!/usr/bin/env python
"""Build a scored RRUFF experimental-PXRD eval set.

RRUFF publishes, per specimen, a measured powder profile (``XY_Processed``) and
a DIF file holding the cell refined against that profile plus the asymmetric
unit of the structure from the cited single-crystal determination.

This differs from the COD/opXRD hold-out in one way that must be stated
wherever these numbers are reported: **the atomic coordinates are
literature-derived, not refined from the same powder measurement.** Only the
cell comes from the specimen. The published RRUFF benchmarks use exactly this
convention, so the set is comparable to them, but it is a weaker ground truth
than a same-specimen refinement and it is why the earlier
``public_exp_pxrd_v1`` protocol excluded RRUFF outright.

Every structure is admitted only if it reproduces the reflection list carried
in its own DIF file, which is direct evidence that the symmetry expansion and
element assignment are faithful.

Primitive-cell site counts are recorded rather than filtered, so the
``<= 20 atoms`` slice used by published RRUFF benchmarks can be taken
downstream without rebuilding.

Usage::

  python data_prep/official_multisource/scripts/build_rruff_exp_pxrd.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

GRID_MIN = 5.0
GRID_MAX = 80.0
REF_WAVELENGTH = 1.54184
MAX_SITES = 80
MIN_PEAKS = 8
COSINE_MIN = 0.40
# Agreement between PXRD simulated from the rebuilt structure and the
# reflection list the DIF file itself carries.
DIF_SELFCHECK_MIN = 0.90

# RRUFF curator identification status. Entries whose identity was never
# confirmed by a diffraction measurement cannot anchor a structural ground
# truth, so they are not scored.
STATUS_REJECT_SUBSTRINGS = (
    "not yet confirmed",
    "confirmed only by chemical analysis",
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "rruff_exp_pxrd"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # @dataclass resolves annotations through sys.modules, so the module has to
    # be registered before it executes.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PUB = _load(Path(__file__).with_name("build_public_exp_pxrd.py"), "_pub_builder")
DIF = _load(Path(__file__).with_name("rruff_dif.py"), "_rruff_dif")
collab = PUB.collab
npd = PUB.npd


PROTOCOL: dict[str, Any] = {
    "name": "rruff_exp_pxrd_v1",
    "source": "RRUFF powder XRD (XY_Processed) + RRUFF DIF structures",
    "pxrd": "experimental, Q-aligned to lambda=1.54184 A on 5-80 deg / 0.05 deg",
    "neutron_xrf_xps": "simulated from the paired structure",
    "ground_truth_caveat": (
        "Cell parameters are refined from this specimen's powder pattern; atomic "
        "coordinates come from the single-crystal determination cited in the DIF "
        "file, not from the same powder measurement."
    ),
    "cell_convention": (
        "primitive; the DIF gives the space-group-expanded conventional cell, "
        "reduced here to match how the model was trained and how the matcher scores"
    ),
    "scored_if": {
        "dif_selfcheck_cosine": f">={DIF_SELFCHECK_MIN}",
        "primitive_n_sites": [2, MAX_SITES],
        "n_pxrd_bins": f">={MIN_PEAKS}",
        "simulated_pxrd_cosine": f">={COSINE_MIN}",
        "curator_status_not": list(STATUS_REJECT_SUBSTRINGS),
        "tokenizer_legal_elements": True,
    },
    "dedup": "reduced_formula + space_group, keep highest simulated_pxrd_cosine",
}


def parse_xy(path: Path) -> dict[str, Any] | None:
    header: dict[str, str] = {}
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            if "=" in line:
                k, v = line[2:].split("=", 1)
                header[k.strip()] = v.strip()
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    if len(xs) < 50:
        return None
    return {
        "header": header,
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "rruff_id": DIF.rruff_id_from_name(path) or path.stem,
    }


def encode_profile_peaks(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_step: float = 0.05,
    rel_threshold: float = 0.01,
    min_separation_deg: float = 0.15,
    max_peaks: int = 256,
) -> tuple[list[float], list[float], int]:
    """Bin, baseline-subtract and pick Bragg maxima from a dense profile.

    The shared ``encode_profile`` keeps every grid bin above a relative
    threshold, which is adequate for the sparse COD/opXRD patterns but turns a
    continuously sampled RRUFF profile into ~1100 "peaks" that trace the
    envelope rather than the reflections. Local-maximum detection recovers an
    actual reflection list, which is what both the rerank and the peak-list
    front-end expect.
    """
    from scipy.signal import find_peaks

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    mask = (xs >= GRID_MIN) & (xs <= GRID_MAX)
    xs, ys = xs[mask], ys[mask]
    if xs.size == 0:
        return [], [], 0
    n = int(round((GRID_MAX - GRID_MIN) / grid_step)) + 1
    grid = GRID_MIN + np.arange(n) * grid_step
    acc = np.zeros(n, dtype=float)
    idx = np.round((xs - GRID_MIN) / grid_step).astype(int)
    keep = (idx >= 0) & (idx < n)
    for i, y in zip(idx[keep], ys[keep]):
        if y > acc[i]:
            acc[i] = float(y)

    bkg = collab._rolling_percentile(acc, window=max(21, n // 25), q=15.0)
    net = np.clip(acc - bkg, 0.0, None)
    peak_max = float(net.max())
    if peak_max <= 0:
        return [], [], int(xs.size)
    net = net / peak_max

    distance = max(1, int(round(min_separation_deg / grid_step)))
    pos, props = find_peaks(
        net, height=rel_threshold, distance=distance, prominence=rel_threshold
    )
    if pos.size == 0:
        return [], [], int(xs.size)
    heights = net[pos]
    if pos.size > max_peaks:
        keep_idx = np.argsort(heights)[::-1][:max_peaks]
        keep_idx.sort()
        pos, heights = pos[keep_idx], heights[keep_idx]
    return grid[pos].tolist(), heights.tolist(), int(xs.size)


def status_rejected(status: str) -> bool:
    low = (status or "").lower()
    return any(s in low for s in STATUS_REJECT_SUBSTRINGS)


def build_index(dif_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(dif_dir.glob("*.txt")):
        rid = DIF.rruff_id_from_name(p)
        if rid:
            out.setdefault(rid, p)
    return out


def process_entry(
    xy_path: Path,
    dif_path: Path,
    legal_elements: set[str],
    max_sites: int,
) -> dict[str, Any]:
    rid = DIF.rruff_id_from_name(xy_path) or xy_path.stem
    status: dict[str, Any] = {
        "id": f"rruff:{rid}",
        "source": "rruff",
        "status": "failed",
        "reason": None,
        "record": None,
    }

    xy = parse_xy(xy_path)
    if xy is None:
        status["reason"] = "no_profile"
        return status
    header = xy["header"]
    status["mineral"] = header.get("NAMES")
    status["curator_status"] = header.get("STATUS")
    status["ideal_chemistry"] = header.get("IDEAL CHEMISTRY")
    if status_rejected(header.get("STATUS", "")):
        status["reason"] = "curator_status_unconfirmed"
        return status

    rec = DIF.parse_dif(dif_path)
    if rec is None:
        status["reason"] = "dif_parse_failed"
        return status
    if not rec.has_atoms:
        status["reason"] = "dif_no_atoms"
        return status
    structure, info = DIF.structure_from_dif(rec)
    if structure is None:
        status["reason"] = str(info)
        return status

    wavelength = float(rec.wavelength or REF_WAVELENGTH)
    selfcheck = DIF.pattern_cosine(structure, rec.reflections, wavelength)
    status["dif_selfcheck_cosine"] = selfcheck
    if selfcheck is None:
        status["reason"] = "dif_no_reflections"
        return status
    if selfcheck < DIF_SELFCHECK_MIN:
        status["reason"] = f"dif_selfcheck_low:{selfcheck:.3f}"
        return status

    # The DIF gives the space-group-expanded conventional cell, but the model
    # was trained on primitive cells and the matcher reduces to primitive
    # anyway, so the primitive cell is the target that keeps eval aligned with
    # training and with the published RRUFF benchmarks.
    conventional_n_sites = int(len(structure))
    try:
        structure = structure.get_primitive_structure()
    except Exception:
        pass
    n_sites = int(len(structure))
    status["conventional_n_sites"] = conventional_n_sites
    status["primitive_n_sites"] = n_sites
    status["n_sites"] = n_sites
    if n_sites < 2:
        status["reason"] = "too_few_sites"
        return status
    if n_sites > max_sites:
        status["reason"] = f"too_many_sites:{n_sites}"
        return status

    atom_types = [npd.site_symbol(site) for site in structure]
    illegal = PUB.tokenizer_illegal(atom_types, legal_elements)
    if illegal:
        status["reason"] = f"tokenizer_illegal:{illegal}"
        return status

    pattern = {
        "x": xy["x"],
        "y": xy["y"],
        "wavelength": wavelength,
        "kind": "profile",
    }
    try:
        xs, ys, q_aligned = npd.remap_two_theta_to_train(
            np.asarray(xy["x"], dtype=float),
            np.asarray(xy["y"], dtype=float),
            wavelength,
            REF_WAVELENGTH,
        )
        peak_x, peak_y, n_profile = encode_profile_peaks(xs, ys)
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"encode_failed:{type(exc).__name__}"
        return status
    if len(peak_x) < MIN_PEAKS:
        status["reason"] = f"too_few_pxrd_bins:{len(peak_x)}"
        return status

    row = {
        "formula": header.get("IDEAL CHEMISTRY"),
        "doi": None,
        "journal": None,
        "year": None,
        "title": rec.reference or header.get("NAMES"),
        "sgNumber": None,
    }
    try:
        record = PUB.to_native_record(
            sample_id=f"rruff:{rid}",
            source="rruff_experimental_pxrd",
            source_id=rid,
            row=row,
            structure=structure,
            peak_x=peak_x,
            peak_y=peak_y,
            pattern_meta=pattern,
            n_profile=n_profile,
            q_aligned=q_aligned,
            disordered=False,
        )
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"native_failed:{type(exc).__name__}"
        return status

    record["experimental"].update(
        {
            "rruff_id": rid,
            "mineral": header.get("NAMES"),
            "locality": header.get("LOCALITY"),
            "curator_status": header.get("STATUS"),
            "dif_selfcheck_cosine": selfcheck,
            "dif_spacegroup_symbol": rec.spacegroup,
            "dif_reference": rec.reference,
            "primitive_n_sites": status["primitive_n_sites"],
            "conventional_n_sites": status["conventional_n_sites"],
            "cell_convention": "primitive",
            "coordinates_provenance": "literature single-crystal (DIF citation)",
        }
    )

    status.update(
        {
            "status": "ok",
            "record": record,
            "formula": record["formula"],
            "space_group": record["space_group"],
            "n_pxrd_bins": len(peak_x),
            "simulated_pxrd_cosine": record["experimental"]["simulated_pxrd_cosine"],
        }
    )
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--max-sites", type=int, default=MAX_SITES)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))

    legal_elements = PUB.load_tokenizer_elements(PUB.TOKENIZER_PATH)
    dif_index = build_index(raw_dir / "dif")
    xy_paths = sorted((raw_dir / "xy").glob("*.txt"))
    if args.limit:
        xy_paths = xy_paths[: args.limit]

    results: list[dict[str, Any]] = []
    for i, xy_path in enumerate(xy_paths, 1):
        rid = DIF.rruff_id_from_name(xy_path)
        dif_path = dif_index.get(rid) if rid else None
        if dif_path is None:
            results.append(
                {
                    "id": f"rruff:{rid or xy_path.stem}",
                    "source": "rruff",
                    "status": "failed",
                    "reason": "no_dif_for_id",
                    "record": None,
                }
            )
            continue
        results.append(process_entry(xy_path, dif_path, legal_elements, args.max_sites))
        if i % 100 == 0:
            ok = sum(1 for r in results if r["status"] == "ok")
            print(f"  {i}/{len(xy_paths)} processed, {ok} ok", flush=True)

    for row in results:
        row["scored"] = bool(row["status"] == "ok" and PUB.scored_mask(row))

    scored_rows = [r for r in results if r["scored"]]
    unique_rows = PUB.dedup_rows(scored_rows)
    unique_ids = {r["id"] for r in unique_rows}
    for row in results:
        row["scored_unique"] = bool(row["id"] in unique_ids)

    def dump(rows: list[dict[str, Any]], path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r["record"], ensure_ascii=False) + "\n")
        return len(rows)

    ok_rows = [r for r in results if r["status"] == "ok"]
    n_all = dump(ok_rows, out_dir / "native" / "test.jsonl")
    n_scored = dump(scored_rows, out_dir / "native_scored" / "test.jsonl")
    n_unique = dump(unique_rows, out_dir / "native_scored_unique" / "test.jsonl")
    n_seq, _ = PUB.write_sequences(
        [r["record"] for r in unique_rows],
        out_dir / "crystal_language_v2" / "test.jsonl",
    )

    fields = [
        "id",
        "source",
        "status",
        "reason",
        "mineral",
        "curator_status",
        "formula",
        "n_sites",
        "primitive_n_sites",
        "space_group",
        "n_pxrd_bins",
        "dif_selfcheck_cosine",
        "simulated_pxrd_cosine",
        "scored",
        "scored_unique",
    ]
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    prim = [
        r.get("primitive_n_sites")
        for r in unique_rows
        if r.get("primitive_n_sites")
    ]
    summary = {
        "protocol": PROTOCOL["name"],
        "attempted": len(results),
        "ok": n_all,
        "scored": n_scored,
        "scored_unique": n_unique,
        "sequences": n_seq,
        "skip_reasons": Counter(
            str(r["reason"]).split(":")[0] for r in results if r["status"] != "ok"
        ).most_common(15),
        "primitive_n_sites_le_20": int(sum(1 for v in prim if v <= 20)),
        "primitive_n_sites_median": float(np.median(prim)) if prim else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
