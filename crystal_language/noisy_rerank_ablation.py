#!/usr/bin/env python
"""Noisy-aware rerank ablation on a frozen candidate pool.

Two modes (no re-generation):

  weights   Re-score from stored ``candidates[*].rerank`` terms.  Seconds.
  spectrum  Rebuild noisy reference spectra (same seed as eval), simulate
            candidate PXRD/ND peaks once, then sweep match sigma / 2θ shift /
            dense-vs-peak reference / weights.  CPU-heavy; caches peaks.

Usage
  # weight sweep on the §8.11 noisy ladder
  python crystal_language/noisy_rerank_ablation.py weights \\
    --eval-dir crystal_language/eval_results/ladder_tierA_s2_noisy_exp_500 \\
    --out crystal_language/eval_results/noisy_rerank_ablation_weights.json

  # spectrum matching (focused lattice-free rungs)
  python crystal_language/noisy_rerank_ablation.py spectrum \\
    --eval-dir crystal_language/eval_results/ladder_tierA_s2_noisy_exp_500 \\
    --data-dir data_prep/outputs/crystal_language_unified_tierA_v2 \\
    --conditions pxrd_xrf,pxrd_neutron_xrf,pxrd_formula \\
    --lattices 0 --workers 16 \\
    --out crystal_language/eval_results/noisy_rerank_ablation_spectrum.json
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
# Running as ``python crystal_language/foo.py`` puts SCRIPT_DIR on sys.path[0],
# which shadows the package name. Prefer the repo root.
sys.path = [str(PACKAGE_ROOT)] + [
    p for p in sys.path if Path(p).resolve() != SCRIPT_DIR
]

from crystal_language.evaluate_generation import peaks_from_dense_spectrum  # noqa: E402
from crystal_language.generate import build_prompt_v2  # noqa: E402
from crystal_language.noise import rasterize_noisy_spectrum  # noqa: E402
from crystal_language.rerank import (  # noqa: E402
    GRID_MAX,
    GRID_MIN,
    GRID_STEP,
    INVALID_PENALTY,
    MAX_PEAKS,
    SIGMA,
    W_CHARGE,
    W_ND,
    W_PXRD,
    rasterize_peaks,
    spectrum_cosine,
    structure_dedup_key,
)

RUNGS = [
    "pxrd",
    "pxrd_element",
    "pxrd_xrf",
    "pxrd_neutron",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
]
NEUTRON_RUNGS = {"pxrd_neutron", "pxrd_neutron_xrf", "pxrd_neutron_xrf_xps"}
TOP_KS = [1, 5, 10, 20, 50]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def topk_any(flags: list[bool], k: int) -> bool:
    return any(flags[:k])


def metrics_from_flags(all_flags: list[list[bool]]) -> dict[str, float]:
    n = max(1, len(all_flags))
    out = {f"T{k}": sum(int(topk_any(f, k)) for f in all_flags) / n for k in TOP_KS}
    out["N"] = float(len(all_flags))
    return out


def order_by_score(
    compact: list[dict[str, Any]],
    score_fn,
) -> list[bool]:
    valid = [i for i, c in enumerate(compact) if c.get("scorable")]
    invalid = [i for i, c in enumerate(compact) if not c.get("scorable")]
    groups: dict[tuple, list[int]] = {}
    for i in valid:
        groups.setdefault(compact[i]["key"], []).append(i)

    score: dict[int, float] = {}
    for members in groups.values():
        s = float(score_fn(compact[members[0]]))
        for m in members:
            score[m] = s
    reps = [members[0] for members in groups.values()]
    uniq = sorted(reps, key=lambda i: (-(score[i] or float("-inf")), i))
    dup = sorted(
        [m for members in groups.values() for m in members[1:]],
        key=lambda i: (-(score[i] or float("-inf")), i),
    )
    order = uniq + dup + invalid
    return [bool(compact[i]["sm"]) for i in order]


# ---------------------------------------------------------------------------
# weights mode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightPolicy:
    label: str
    w_pxrd: float = W_PXRD
    w_nd: float = W_ND
    w_charge: float = W_CHARGE
    invalid_penalty: float = INVALID_PENALTY
    cos_power: float = 1.0  # >1 sharpens high-cos discrimination


WEIGHT_POLICIES = [
    WeightPolicy("baseline (clean defaults)"),
    WeightPolicy("w_nd=0", w_nd=0.0),
    WeightPolicy("w_nd=0.1", w_nd=0.1),
    WeightPolicy("w_nd=0.5", w_nd=0.5),
    WeightPolicy("w_nd=1.0", w_nd=1.0),
    WeightPolicy("w_charge=0", w_charge=0.0),
    WeightPolicy("w_charge=0.5", w_charge=0.5),
    WeightPolicy("invalid=-1", invalid_penalty=-1.0),
    WeightPolicy("cos^2", cos_power=2.0),
    WeightPolicy("cos^3", cos_power=3.0),
    WeightPolicy("noisy-A: nd0.1+cos^2", w_nd=0.1, cos_power=2.0),
    WeightPolicy("noisy-B: nd0.5+cos^2", w_nd=0.5, cos_power=2.0),
    WeightPolicy("noisy-C: charge0.5+cos^2", w_charge=0.5, cos_power=2.0),
    WeightPolicy("noisy-D: nd0.1+charge0.5+cos^2", w_nd=0.1, w_charge=0.5, cos_power=2.0),
]


def load_weight_records(eval_dir: Path, split: str, cond: str, lat: int) -> list[list[dict]]:
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr1_start*_n*_s*.jsonl"
    paths = [p for p in globlib.glob(str(eval_dir / pattern)) if "_summary" not in p]
    paths.sort(
        key=lambda p: int(re.search(r"start(\d+)", p).group(1))
        if re.search(r"start(\d+)", p)
        else 0
    )
    seen: set[Any] = set()
    records: list[list[dict]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "target_error" in row:
                    continue
                idx = row.get("index")
                if idx is not None:
                    if idx in seen:
                        continue
                    seen.add(idx)
                compact: list[dict] = []
                for c in row.get("candidates") or []:
                    d = c.get("rerank")
                    if c.get("structure") is not None and d:
                        compact.append(
                            {
                                "scorable": True,
                                "sm": bool(c.get("structure_match")),
                                "pxrd": d.get("pxrd_cos"),
                                "nd": d.get("nd_cos"),
                                "charge": bool(d.get("charge_ok")),
                                "ok": bool(d.get("valid")) and not d.get("sim_failed"),
                                "key": structure_dedup_key(c["structure"]),
                            }
                        )
                    else:
                        compact.append({"scorable": False, "sm": bool(c.get("structure_match"))})
                records.append(compact)
    return records


def score_weight(c: dict, pol: WeightPolicy, neutron_on: bool) -> float:
    s = 0.0
    if c["pxrd"] is not None:
        s += pol.w_pxrd * (float(c["pxrd"]) ** pol.cos_power)
    if neutron_on and c["nd"] is not None:
        s += pol.w_nd * (float(c["nd"]) ** pol.cos_power)
    s += pol.w_charge * float(c["charge"])
    if not c["ok"]:
        s += pol.invalid_penalty
    return s


def run_weights(args: argparse.Namespace) -> dict:
    eval_dir = Path(args.eval_dir)
    data: dict[tuple[str, int], list[list[dict]]] = {}
    for lat in (0, 1):
        for cond in RUNGS:
            data[(cond, lat)] = load_weight_records(eval_dir, args.split, cond, lat)

    table: list[dict] = []
    for pol in WEIGHT_POLICIES:
        for lat, lat_name in ((0, "lattice_free"), (1, "known_lattice")):
            for cond in RUNGS:
                neutron_on = cond in NEUTRON_RUNGS and pol.w_nd != 0.0
                recs = data[(cond, lat)]
                flags = [
                    order_by_score(r, lambda c, p=pol, n=neutron_on: score_weight(c, p, n))
                    for r in recs
                ]
                m = metrics_from_flags(flags)
                row = {
                    "policy": pol.label,
                    "lattice": lat_name,
                    "condition": cond,
                    **{k: round(100.0 * v, 2) if k.startswith("T") else int(v) for k, v in m.items()},
                    "params": asdict(pol),
                }
                table.append(row)

    # Focused summary: lattice-free high-info rungs vs baseline T1.
    focus = ["pxrd_element", "pxrd_xrf", "pxrd_neutron_xrf", "pxrd_neutron_xrf_xps", "pxrd_formula"]
    baseline = {
        (r["condition"], r["lattice"]): r
        for r in table
        if r["policy"] == "baseline (clean defaults)"
    }
    summary_rows = []
    for pol_label in [p.label for p in WEIGHT_POLICIES]:
        deltas = []
        for cond in focus:
            cur = next(
                r
                for r in table
                if r["policy"] == pol_label and r["lattice"] == "lattice_free" and r["condition"] == cond
            )
            base = baseline[(cond, "lattice_free")]
            deltas.append(cur["T1"] - base["T1"])
        summary_rows.append(
            {
                "policy": pol_label,
                "mean_dT1_focus": round(sum(deltas) / len(deltas), 2),
                "dT1_by_cond": {c: round(d, 2) for c, d in zip(focus, deltas)},
            }
        )
    summary_rows.sort(key=lambda r: -r["mean_dT1_focus"])

    result = {
        "mode": "weights",
        "eval_dir": str(eval_dir),
        "baseline_defaults": {
            "W_PXRD": W_PXRD,
            "W_ND": W_ND,
            "W_CHARGE": W_CHARGE,
            "INVALID_PENALTY": INVALID_PENALTY,
            "SIGMA": SIGMA,
        },
        "focus_rungs": focus,
        "summary_lattice_free_focus": summary_rows,
        "table": table,
    }
    return result


# ---------------------------------------------------------------------------
# spectrum mode
# ---------------------------------------------------------------------------

def _simulate_peaks(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: return candidate PXRD/ND peak lists (no scoring)."""
    from pymatgen.core import Lattice, Structure

    out: dict[str, Any] = {"ok": False, "pxrd": None, "nd": None, "sim_failed": True}
    try:
        lat = payload["lattice"]
        structure = Structure(
            Lattice.from_parameters(
                lat["a"], lat["b"], lat["c"], lat["alpha"], lat["beta"], lat["gamma"]
            ),
            payload["species"],
            payload["frac_coords"],
        )
        n = len(structure)
        vol_per_atom = structure.volume / max(1, n)
        min_dist = float("inf")
        if n > 1:
            dm = structure.distance_matrix
            min_dist = float(np.min(dm[np.triu_indices(n, k=1)]))
        valid = 2.0 <= vol_per_atom <= 150.0 and (n == 1 or min_dist >= 0.5)
        out["ok"] = bool(valid)
        if not valid:
            out["sim_failed"] = False
            return out
    except Exception:
        return out

    def top_peaks(x, y):
        if len(x) == 0:
            return []
        ymax = float(np.max(y))
        if ymax <= 0:
            return []
        order = np.argsort(-y)[:MAX_PEAKS]
        return [(float(x[j]), float(y[j] / ymax)) for j in sorted(order, key=lambda j: x[j])]

    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator

        pattern = XRDCalculator().get_pattern(
            structure, scaled=True, two_theta_range=(GRID_MIN, GRID_MAX)
        )
        out["pxrd"] = top_peaks(np.asarray(pattern.x), np.asarray(pattern.y))
        out["sim_failed"] = False
    except Exception:
        out["sim_failed"] = True
        return out
    try:
        from pymatgen.analysis.diffraction.neutron import NDCalculator

        pattern = NDCalculator(wavelength=1.54184).get_pattern(
            structure, scaled=True, two_theta_range=(GRID_MIN, GRID_MAX)
        )
        out["nd"] = top_peaks(np.asarray(pattern.x), np.asarray(pattern.y))
    except Exception:
        out["nd"] = None
    return out


def _key_to_str(key: tuple) -> str:
    return json.dumps(key, ensure_ascii=False)


def load_source_records(data_dir: Path, split: str, indices: set[int]) -> dict[int, dict]:
    path = data_dir / f"{split}.jsonl"
    want = set(indices)
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i not in want:
                continue
            rec = json.loads(line)
            rec["_source_index"] = i
            out[i] = rec
            if len(out) >= len(want):
                break
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"missing {len(missing)} source indices, e.g. {sorted(missing)[:5]}")
    return out


def rebuild_noisy_refs(
    record: dict,
    condition_set: str,
    use_lattice: bool,
    noise_seed: int,
    spectra_slots: int = 16,
) -> dict[str, Any]:
    """Reproduce eval-time noisy dense spectra / peak refs for one record."""
    import torch

    rng = random.Random(noise_seed + int(record["_source_index"]))
    _, used = build_prompt_v2(
        record,
        condition_set,
        use_lattice,
        spectra_slots,
        rng=rng,
        noise_profile="experimental",
    )
    spec = record.get("spectra") or {}
    dense: dict[str, np.ndarray | None] = {"pxrd": None, "neutron": None}
    peaks: dict[str, list[tuple[float, float]] | None] = {"pxrd": None, "neutron": None}
    for name in ("pxrd", "neutron"):
        if not used.get(name):
            continue
        vec = rasterize_noisy_spectrum(
            spec.get(f"{name}_x") or [],
            spec.get(f"{name}_y") or [],
            name=name,
            rng=rng,
            profile="experimental",
        )
        dense[name] = vec.detach().cpu().numpy().astype(np.float64)
        peaks[name] = peaks_from_dense_spectrum(torch.as_tensor(dense[name]))
    return {"dense": dense, "peaks": peaks, "used": used}


def cosine_with_shift(
    cand_peaks: list[tuple[float, float]] | None,
    ref_peaks: list[tuple[float, float]] | None,
    *,
    sigma: float,
    shift_max: float,
    shift_step: float,
) -> float | None:
    if not cand_peaks or not ref_peaks:
        return None
    cand = rasterize_peaks(cand_peaks, sigma=sigma)
    if shift_max <= 0:
        return spectrum_cosine(cand, rasterize_peaks(ref_peaks, sigma=sigma))
    best = 0.0
    n_step = int(round(shift_max / shift_step))
    for k in range(-n_step, n_step + 1):
        shift = k * shift_step
        shifted = [(t + shift, i) for t, i in ref_peaks]
        best = max(best, spectrum_cosine(cand, rasterize_peaks(shifted, sigma=sigma)))
    return best


def cosine_dense_with_shift(
    cand_peaks: list[tuple[float, float]] | None,
    ref_dense: np.ndarray | None,
    *,
    sigma: float,
    shift_max: float,
    shift_step: float,
) -> float | None:
    if not cand_peaks or ref_dense is None:
        return None
    cand = rasterize_peaks(cand_peaks, sigma=sigma)
    if shift_max <= 0:
        return spectrum_cosine(cand, ref_dense)
    grid = np.arange(GRID_MIN, GRID_MAX + GRID_STEP, GRID_STEP)
    best = 0.0
    n_step = int(round(shift_max / shift_step))
    for k in range(-n_step, n_step + 1):
        shift = k * shift_step
        # shift dense ref by interpolating on the 2θ grid
        shifted = np.interp(grid, grid + shift, ref_dense, left=0.0, right=0.0)
        best = max(best, spectrum_cosine(cand, shifted))
    return best


@dataclass(frozen=True)
class SpectrumPolicy:
    label: str
    sigma: float = SIGMA
    shift_max: float = 0.0
    shift_step: float = 0.05
    ref_mode: str = "peaks"  # peaks | dense
    w_pxrd: float = W_PXRD
    w_nd: float = W_ND
    w_charge: float = W_CHARGE
    invalid_penalty: float = INVALID_PENALTY
    cos_power: float = 1.0


SPECTRUM_POLICIES = [
    SpectrumPolicy("baseline peaks σ=0.25"),
    SpectrumPolicy("peaks σ=0.40", sigma=0.40),
    SpectrumPolicy("peaks σ=0.60", sigma=0.60),
    SpectrumPolicy("peaks σ=0.25 ±0.20°", shift_max=0.20),
    SpectrumPolicy("peaks σ=0.40 ±0.20°", sigma=0.40, shift_max=0.20),
    SpectrumPolicy("peaks σ=0.40 ±0.30°", sigma=0.40, shift_max=0.30),
    SpectrumPolicy("dense σ=0.25", ref_mode="dense"),
    SpectrumPolicy("dense σ=0.40", ref_mode="dense", sigma=0.40),
    SpectrumPolicy("dense σ=0.40 ±0.20°", ref_mode="dense", sigma=0.40, shift_max=0.20),
    SpectrumPolicy(
        "best-guess: dense σ=0.40 ±0.20° + cos^2 + nd0.1",
        ref_mode="dense",
        sigma=0.40,
        shift_max=0.20,
        w_nd=0.1,
        cos_power=2.0,
    ),
    SpectrumPolicy(
        "best-guess: peaks σ=0.40 ±0.20° + cos^2 + nd0.1",
        sigma=0.40,
        shift_max=0.20,
        w_nd=0.1,
        cos_power=2.0,
    ),
]

# Default noisy-aware deployment (§8.11.2).
DEPLOY_POLICIES = [
    SpectrumPolicy("baseline peaks σ=0.25"),
    SpectrumPolicy("deploy peaks σ=0.40 ±0.20°", sigma=0.40, shift_max=0.20),
]


def load_spectrum_pool(
    eval_dir: Path,
    split: str,
    cond: str,
    lat: int,
) -> list[dict]:
    """Load records with structures for spectrum re-score."""
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr1_start*_n*_s*.jsonl"
    paths = [p for p in globlib.glob(str(eval_dir / pattern)) if "_summary" not in p]
    paths.sort(
        key=lambda p: int(re.search(r"start(\d+)", p).group(1))
        if re.search(r"start(\d+)", p)
        else 0
    )
    seen: set[Any] = set()
    records: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "target_error" in row:
                    continue
                idx = row.get("index")
                if idx is not None:
                    if idx in seen:
                        continue
                    seen.add(idx)
                cands = []
                for c in row.get("candidates") or []:
                    d = c.get("rerank")
                    if c.get("structure") is not None and d:
                        key = structure_dedup_key(c["structure"])
                        cands.append(
                            {
                                "scorable": True,
                                "sm": bool(c.get("structure_match")),
                                "charge": bool(d.get("charge_ok")),
                                "ok_stored": bool(d.get("valid")) and not d.get("sim_failed"),
                                "pxrd_stored": d.get("pxrd_cos"),
                                "nd_stored": d.get("nd_cos"),
                                "key": key,
                                "key_str": _key_to_str(key),
                                "structure": c["structure"],
                            }
                        )
                    else:
                        cands.append({"scorable": False, "sm": bool(c.get("structure_match"))})
                records.append({"index": int(idx), "id": row.get("id"), "candidates": cands})
    return records


def run_spectrum(args: argparse.Namespace) -> dict:
    eval_dir = Path(args.eval_dir)
    data_dir = Path(args.data_dir)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    lattices = [int(x) for x in args.lattices.split(",") if x.strip() != ""]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    peak_cache_path = cache_dir / "candidate_peaks.jsonl"
    peak_cache: dict[str, dict] = {}
    if peak_cache_path.exists():
        with peak_cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                peak_cache[row["key"]] = row

    # Discover noise seed from any summary.
    noise_seed = args.noise_seed
    summaries = list(eval_dir.glob("*_summary.json"))
    if summaries and noise_seed is None:
        meta = json.loads(summaries[0].read_text())
        noise_seed = int(meta.get("noise_seed", 20260706))
    if noise_seed is None:
        noise_seed = 20260706

    all_results: list[dict] = []
    sanity: list[dict] = []

    for lat in lattices:
        for cond in conditions:
            print(f"=== spectrum ablation: {cond} lat={lat} ===", flush=True)
            pool = load_spectrum_pool(eval_dir, args.split, cond, lat)
            if args.max_records is not None:
                pool = pool[: args.max_records]
            indices = {r["index"] for r in pool}
            source = load_source_records(data_dir, args.split, indices)

            # Collect unique structures needing simulation.
            need: dict[str, dict] = {}
            for rec in pool:
                for c in rec["candidates"]:
                    if not c.get("scorable"):
                        continue
                    ks = c["key_str"]
                    if ks not in peak_cache and ks not in need:
                        need[ks] = c["structure"]
            print(f"  records={len(pool)} unique_to_sim={len(need)} cached={len(peak_cache)}", flush=True)

            if need:
                payloads = [{"key": k, **v} for k, v in need.items()]
                # strip key from structure payload for worker
                jobs = []
                keys = []
                for item in payloads:
                    keys.append(item["key"])
                    jobs.append(
                        {
                            "lattice": item["lattice"],
                            "species": item["species"],
                            "frac_coords": item["frac_coords"],
                        }
                    )
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    sims = list(ex.map(_simulate_peaks, jobs, chunksize=8))
                with peak_cache_path.open("a", encoding="utf-8") as fh:
                    for ks, sim in zip(keys, sims):
                        row = {"key": ks, **sim}
                        peak_cache[ks] = row
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  simulated {len(keys)} structures", flush=True)

            # Rebuild refs + attach peaks to candidates.
            prepared: list[dict] = []
            for rec in pool:
                src = source[rec["index"]]
                refs = rebuild_noisy_refs(
                    src,
                    condition_set=cond,
                    use_lattice=bool(lat),
                    noise_seed=noise_seed,
                )
                compact = []
                for c in rec["candidates"]:
                    if not c.get("scorable"):
                        compact.append({"scorable": False, "sm": c["sm"]})
                        continue
                    sim = peak_cache[c["key_str"]]
                    compact.append(
                        {
                            "scorable": True,
                            "sm": c["sm"],
                            "charge": c["charge"],
                            "ok": bool(sim.get("ok")) and not sim.get("sim_failed"),
                            "key": c["key"],
                            "pxrd_peaks": sim.get("pxrd"),
                            "nd_peaks": sim.get("nd"),
                            "pxrd_stored": c.get("pxrd_stored"),
                            "nd_stored": c.get("nd_stored"),
                            "ref_peaks": refs["peaks"],
                            "ref_dense": refs["dense"],
                        }
                    )
                prepared.append(compact)

                # Sanity: compare recomputed baseline cosine vs stored for first few.
                if len(sanity) < 30:
                    for c in compact:
                        if not c.get("scorable") or c.get("pxrd_stored") is None:
                            continue
                        re_cos = cosine_with_shift(
                            c["pxrd_peaks"],
                            c["ref_peaks"]["pxrd"],
                            sigma=SIGMA,
                            shift_max=0.0,
                            shift_step=0.05,
                        )
                        if re_cos is None:
                            continue
                        sanity.append(
                            {
                                "condition": cond,
                                "index": rec["index"],
                                "stored": c["pxrd_stored"],
                                "recomputed": round(re_cos, 4),
                                "abs_diff": round(abs(re_cos - float(c["pxrd_stored"])), 4),
                            }
                        )
                        break

            policy_bank = DEPLOY_POLICIES if args.policy_set == "deploy" else SPECTRUM_POLICIES
            for pol in policy_bank:
                neutron_on = cond in NEUTRON_RUNGS and pol.w_nd != 0.0

                def make_score(p=pol, n_on=neutron_on):
                    def _score(c: dict) -> float:
                        if p.ref_mode == "dense":
                            px = cosine_dense_with_shift(
                                c["pxrd_peaks"],
                                c["ref_dense"]["pxrd"],
                                sigma=p.sigma,
                                shift_max=p.shift_max,
                                shift_step=p.shift_step,
                            )
                            nd = (
                                cosine_dense_with_shift(
                                    c["nd_peaks"],
                                    c["ref_dense"]["neutron"],
                                    sigma=p.sigma,
                                    shift_max=p.shift_max,
                                    shift_step=p.shift_step,
                                )
                                if n_on
                                else None
                            )
                        else:
                            px = cosine_with_shift(
                                c["pxrd_peaks"],
                                c["ref_peaks"]["pxrd"],
                                sigma=p.sigma,
                                shift_max=p.shift_max,
                                shift_step=p.shift_step,
                            )
                            nd = (
                                cosine_with_shift(
                                    c["nd_peaks"],
                                    c["ref_peaks"]["neutron"],
                                    sigma=p.sigma,
                                    shift_max=p.shift_max,
                                    shift_step=p.shift_step,
                                )
                                if n_on
                                else None
                            )
                        s = 0.0
                        if px is not None:
                            s += p.w_pxrd * (float(px) ** p.cos_power)
                        if nd is not None:
                            s += p.w_nd * (float(nd) ** p.cos_power)
                        s += p.w_charge * float(c["charge"])
                        if not c["ok"]:
                            s += p.invalid_penalty
                        return s

                    return _score

                flags = [order_by_score(r, make_score()) for r in prepared]
                m = metrics_from_flags(flags)
                all_results.append(
                    {
                        "policy": pol.label,
                        "lattice": "lattice_free" if lat == 0 else "known_lattice",
                        "condition": cond,
                        **{
                            k: round(100.0 * v, 2) if k.startswith("T") else int(v)
                            for k, v in m.items()
                        },
                        "params": asdict(pol),
                    }
                )
                print(
                    f"  {pol.label:<48} T1={100*m['T1']:5.1f} T20={100*m['T20']:5.1f}",
                    flush=True,
                )

    # Rank policies by mean T1 over completed condition/lattice cells.
    by_pol: dict[str, list[float]] = {}
    for r in all_results:
        by_pol.setdefault(r["policy"], []).append(r["T1"])
    ranking = sorted(
        (
            {"policy": k, "mean_T1": round(sum(v) / len(v), 2), "n_cells": len(v)}
            for k, v in by_pol.items()
        ),
        key=lambda x: -x["mean_T1"],
    )

    return {
        "mode": "spectrum",
        "eval_dir": str(eval_dir),
        "data_dir": str(data_dir),
        "noise_seed": noise_seed,
        "conditions": conditions,
        "lattices": lattices,
        "sanity_recompute_vs_stored": {
            "n": len(sanity),
            "mean_abs_diff": round(
                sum(s["abs_diff"] for s in sanity) / max(1, len(sanity)), 4
            ),
            "samples": sanity[:10],
        },
        "ranking_mean_T1": ranking,
        "table": all_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_weight_summary(result: dict) -> None:
    print("\n=== lattice-free focus mean ΔT1 vs baseline ===")
    print(f"{'policy':<42}{'mean_dT1':>10}")
    for row in result["summary_lattice_free_focus"]:
        print(f"{row['policy']:<42}{row['mean_dT1_focus']:>10.2f}")
    best = result["summary_lattice_free_focus"][0]
    print("\nbest policy detail (dT1 by condition):")
    print(json.dumps(best, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    w = sub.add_parser("weights", help="offline weight / score-shape sweep")
    w.add_argument("--eval-dir", required=True)
    w.add_argument("--split", default="test")
    w.add_argument("--out", required=True)

    s = sub.add_parser("spectrum", help="rebuild noisy refs + sweep match geometry")
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--data-dir", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--conditions", default="pxrd_xrf,pxrd_neutron_xrf,pxrd_formula")
    s.add_argument("--lattices", default="0")
    s.add_argument(
        "--policy-set",
        choices=["full", "deploy"],
        default="full",
        help="full=all spectrum sweeps; deploy=baseline vs σ=0.40±0.20 only",
    )
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--max-records", type=int, default=None)
    s.add_argument("--noise-seed", type=int, default=None)
    s.add_argument(
        "--cache-dir",
        default="crystal_language/eval_results/noisy_rerank_ablation_cache",
    )
    s.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.mode == "weights":
        result = run_weights(args)
        print_weight_summary(result)
    else:
        result = run_spectrum(args)
        print("\n=== spectrum policy ranking (mean T1) ===")
        for row in result["ranking_mean_T1"]:
            print(f"{row['policy']:<55}{row['mean_T1']:>7.2f}")
        print("\nsanity:", json.dumps(result["sanity_recompute_vs_stored"], ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
