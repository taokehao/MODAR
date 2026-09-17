"""Rietveld-lite lattice refine for noisy PXRD/ND reranking.

Given a candidate structure and noisy reference peaks, search a small lattice
neighborhood (isotropic scale and/or per-axis length) to maximize spectrum
cosine. Fractional coordinates stay fixed — under experimental zero-shift /
strain the dominant mismatch is cell metric, not site order.

This is the offline / plug-in core for §8.11.4 (M1). It does not need the LM
and can re-score frozen eval JSONL pools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rerank import (
    GRID_MAX,
    GRID_MIN,
    MAX_PEAKS,
    MIN_DIST,
    MIN_VOL_PER_ATOM,
    MAX_VOL_PER_ATOM,
    NOISY_MATCH_SIGMA,
    NOISY_SHIFT_MAX,
    NOISY_SHIFT_STEP,
    W_CHARGE,
    W_ND,
    W_PXRD,
    INVALID_PENALTY,
    best_spectrum_cosine,
    smact_charge_balanced,
    structure_dedup_key,
)


@dataclass(frozen=True)
class RefineConfig:
    """Search neighborhood around the candidate lattice."""

    # Isotropic a,b,c scale grid (inclusive).  ±3% covers typical strain / zero
    # error left after 2θ shift search.
    scale_max: float = 0.03
    n_scales: int = 7
    # Optional per-axis length search after isotropic (fraction of each length).
    axis_max: float = 0.0
    n_axis: int = 5
    match_sigma: float = NOISY_MATCH_SIGMA
    shift_max: float = NOISY_SHIFT_MAX
    shift_step: float = NOISY_SHIFT_STEP
    w_pxrd: float = W_PXRD
    w_nd: float = W_ND
    w_charge: float = W_CHARGE
    invalid_penalty: float = INVALID_PENALTY


@dataclass
class RefineResult:
    score: float
    pxrd_cos: float | None
    nd_cos: float | None
    valid: bool
    charge_ok: bool
    sim_failed: bool
    scale: float
    axis_scales: tuple[float, float, float]
    lattice: dict[str, float]


def _structure_from_payload(payload: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    lat = payload["lattice"]
    lattice = Lattice.from_parameters(
        lat["a"], lat["b"], lat["c"], lat["alpha"], lat["beta"], lat["gamma"]
    )
    return Structure(lattice, payload["species"], payload["frac_coords"])


def _payload_with_lattice(payload: dict[str, Any], lattice: dict[str, float]) -> dict[str, Any]:
    out = dict(payload)
    out["lattice"] = dict(lattice)
    return out


def _top_peaks(x: np.ndarray, y: np.ndarray, max_peaks: int = MAX_PEAKS) -> list[tuple[float, float]]:
    if len(x) == 0:
        return []
    ymax = float(np.max(y))
    if ymax <= 0:
        return []
    order = np.argsort(-y)[:max_peaks]
    return [(float(x[j]), float(y[j] / ymax)) for j in sorted(order, key=lambda j: x[j])]


def _validity(structure) -> bool:
    n = len(structure)
    vol_per_atom = structure.volume / max(1, n)
    if not (MIN_VOL_PER_ATOM <= vol_per_atom <= MAX_VOL_PER_ATOM):
        return False
    if n <= 1:
        return True
    dm = structure.distance_matrix
    min_dist = float(np.min(dm[np.triu_indices(n, k=1)]))
    return min_dist >= MIN_DIST


def _simulate_peaks(structure, want_nd: bool) -> tuple[list[tuple[float, float]] | None, list[tuple[float, float]] | None, bool]:
    """Return (pxrd_peaks, nd_peaks, sim_failed)."""
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator

        pattern = XRDCalculator().get_pattern(
            structure, scaled=True, two_theta_range=(GRID_MIN, GRID_MAX)
        )
        pxrd = _top_peaks(np.asarray(pattern.x), np.asarray(pattern.y))
    except Exception:
        return None, None, True

    nd = None
    if want_nd:
        try:
            from pymatgen.analysis.diffraction.neutron import NDCalculator

            pattern = NDCalculator(wavelength=1.54184).get_pattern(
                structure, scaled=True, two_theta_range=(GRID_MIN, GRID_MAX)
            )
            nd = _top_peaks(np.asarray(pattern.x), np.asarray(pattern.y))
        except Exception:
            nd = None
    return pxrd, nd, False


def _score_peaks(
    pxrd_peaks: list[tuple[float, float]] | None,
    nd_peaks: list[tuple[float, float]] | None,
    ref_pxrd: list[tuple[float, float]] | None,
    ref_nd: list[tuple[float, float]] | None,
    cfg: RefineConfig,
    charge_ok: bool,
    valid: bool,
    sim_failed: bool,
) -> tuple[float, float | None, float | None]:
    pxrd_cos = None
    nd_cos = None
    if ref_pxrd and pxrd_peaks:
        pxrd_cos = best_spectrum_cosine(
            pxrd_peaks,
            ref_pxrd,
            sigma=cfg.match_sigma,
            shift_max=cfg.shift_max,
            shift_step=cfg.shift_step,
        )
    if ref_nd and nd_peaks:
        nd_cos = best_spectrum_cosine(
            nd_peaks,
            ref_nd,
            sigma=cfg.match_sigma,
            shift_max=cfg.shift_max,
            shift_step=cfg.shift_step,
        )
    score = 0.0
    if pxrd_cos is not None:
        score += cfg.w_pxrd * pxrd_cos
    if nd_cos is not None:
        score += cfg.w_nd * nd_cos
    score += cfg.w_charge * float(charge_ok)
    if (not valid) or sim_failed:
        score += cfg.invalid_penalty
    return score, pxrd_cos, nd_cos


def _scaled_lattice(base: dict[str, float], sa: float, sb: float, sc: float) -> dict[str, float]:
    return {
        "a": float(base["a"]) * sa,
        "b": float(base["b"]) * sb,
        "c": float(base["c"]) * sc,
        "alpha": float(base["alpha"]),
        "beta": float(base["beta"]),
        "gamma": float(base["gamma"]),
    }


def refine_candidate(
    payload: dict[str, Any],
    ref_pxrd: list[tuple[float, float]] | None,
    ref_nd: list[tuple[float, float]] | None,
    cfg: RefineConfig | None = None,
) -> RefineResult:
    """Search a small lattice neighborhood; return best physical score."""
    cfg = cfg or RefineConfig()
    base_lat = {k: float(payload["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
    from collections import Counter

    charge_ok = smact_charge_balanced(sorted(Counter(payload["species"]).items()))
    want_nd = ref_nd is not None

    scales = (
        np.linspace(1.0 - cfg.scale_max, 1.0 + cfg.scale_max, cfg.n_scales)
        if cfg.scale_max > 0 and cfg.n_scales > 1
        else np.asarray([1.0])
    )

    best: RefineResult | None = None

    def consider(sa: float, sb: float, sc: float) -> None:
        nonlocal best
        lat = _scaled_lattice(base_lat, sa, sb, sc)
        try:
            structure = _structure_from_payload(_payload_with_lattice(payload, lat))
            valid = _validity(structure)
        except Exception:
            return
        if not valid:
            cand = RefineResult(
                score=cfg.invalid_penalty + cfg.w_charge * float(charge_ok),
                pxrd_cos=None,
                nd_cos=None,
                valid=False,
                charge_ok=charge_ok,
                sim_failed=False,
                scale=float((sa * sb * sc) ** (1.0 / 3.0)),
                axis_scales=(float(sa), float(sb), float(sc)),
                lattice=lat,
            )
        else:
            pxrd, nd, failed = _simulate_peaks(structure, want_nd=want_nd)
            score, pxrd_cos, nd_cos = _score_peaks(
                pxrd, nd, ref_pxrd, ref_nd, cfg, charge_ok, valid, failed
            )
            cand = RefineResult(
                score=score,
                pxrd_cos=None if pxrd_cos is None else float(pxrd_cos),
                nd_cos=None if nd_cos is None else float(nd_cos),
                valid=True,
                charge_ok=charge_ok,
                sim_failed=failed,
                scale=float((sa * sb * sc) ** (1.0 / 3.0)),
                axis_scales=(float(sa), float(sb), float(sc)),
                lattice=lat,
            )
        if best is None or cand.score > best.score:
            best = cand

    for s in scales:
        consider(float(s), float(s), float(s))

    # Optional axis refinement seeded at the best isotropic scale.
    if best is not None and cfg.axis_max > 0 and cfg.n_axis > 1:
        sa0, sb0, sc0 = best.axis_scales
        axis = np.linspace(1.0 - cfg.axis_max, 1.0 + cfg.axis_max, cfg.n_axis)
        for da in axis:
            consider(sa0 * float(da), sb0, sc0)
        sa1, sb1, sc1 = best.axis_scales
        for db in axis:
            consider(sa1, sb1 * float(db), sc1)
        sa2, sb2, sc2 = best.axis_scales
        for dc in axis:
            consider(sa2, sb2, sc2 * float(dc))

    if best is None:
        return RefineResult(
            score=cfg.invalid_penalty,
            pxrd_cos=None,
            nd_cos=None,
            valid=False,
            charge_ok=charge_ok,
            sim_failed=True,
            scale=1.0,
            axis_scales=(1.0, 1.0, 1.0),
            lattice=base_lat,
        )
    return best


def score_without_refine(
    payload: dict[str, Any],
    ref_pxrd: list[tuple[float, float]] | None,
    ref_nd: list[tuple[float, float]] | None,
    cfg: RefineConfig | None = None,
) -> RefineResult:
    """Single-point score at the original lattice (no lattice search)."""
    base = cfg or RefineConfig()
    frozen = RefineConfig(
        scale_max=0.0,
        n_scales=1,
        axis_max=0.0,
        n_axis=1,
        match_sigma=base.match_sigma,
        shift_max=base.shift_max,
        shift_step=base.shift_step,
        w_pxrd=base.w_pxrd,
        w_nd=base.w_nd,
        w_charge=base.w_charge,
        invalid_penalty=base.invalid_penalty,
    )
    return refine_candidate(payload, ref_pxrd, ref_nd, cfg=frozen)


def rerank_with_optional_refine(
    candidates: list[dict[str, Any]],
    ref_pxrd: list[tuple[float, float]] | None,
    ref_nd: list[tuple[float, float]] | None,
    *,
    cfg: RefineConfig,
    refine_top_m: int = 20,
    do_refine: bool = True,
) -> list[bool]:
    """Return structure_match flags in reranked order.

    Each candidate dict needs: structure (or None), structure_match, and optionally
    precomputed charge_ok in rerank detail. Only the top-M unique structures by
    unreined score are refined when ``do_refine`` is True.
    """
    n = len(candidates)
    compact: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        struct = c.get("structure")
        if struct is None:
            compact.append({"idx": i, "scorable": False, "sm": bool(c.get("structure_match"))})
            continue
        key = structure_dedup_key(struct)
        compact.append(
            {
                "idx": i,
                "scorable": True,
                "sm": bool(c.get("structure_match")),
                "key": key,
                "structure": struct,
                "rerank": c.get("rerank") or {},
            }
        )

    # Initial unreined scores (one per unique key).
    groups: dict[Any, list[int]] = {}
    for j, c in enumerate(compact):
        if c["scorable"]:
            groups.setdefault(c["key"], []).append(j)

    init_score: dict[Any, float] = {}
    init_detail: dict[Any, RefineResult] = {}
    for key, members in groups.items():
        payload = compact[members[0]]["structure"]
        res = score_without_refine(payload, ref_pxrd, ref_nd, cfg=cfg)
        init_score[key] = res.score
        init_detail[key] = res

    # Choose top-M unique keys for refine.
    ranked_keys = sorted(init_score.keys(), key=lambda k: (-init_score[k], str(k)))
    refine_keys = set(ranked_keys[: max(0, refine_top_m)]) if do_refine else set()

    final_score: dict[Any, float] = dict(init_score)
    for key in refine_keys:
        payload = compact[groups[key][0]]["structure"]
        res = refine_candidate(payload, ref_pxrd, ref_nd, cfg=cfg)
        final_score[key] = res.score

    # Order: unique reps by score, then dups, then invalid.
    reps = [members[0] for members in groups.values()]
    uniq = sorted(reps, key=lambda j: (-(final_score[compact[j]["key"]] or float("-inf")), j))
    dups = sorted(
        [m for members in groups.values() for m in members[1:]],
        key=lambda j: (-(final_score[compact[j]["key"]] or float("-inf")), j),
    )
    invalid = [j for j, c in enumerate(compact) if not c["scorable"]]
    order = uniq + dups + invalid
    return [compact[j]["sm"] for j in order]


__all__ = [
    "RefineConfig",
    "RefineResult",
    "refine_candidate",
    "rerank_with_optional_refine",
    "score_without_refine",
]
