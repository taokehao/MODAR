#!/usr/bin/env python3
"""Reconstruct the unpublished RealPXRD-Solver experimental preprocessor.

The paper (arXiv:2603.00965, SI Methods) specifies:

* Map every experimental profile to d-spacing, then re-express it on a
  Cu Kα 2θ axis (λ = 1.5406 Å).
* Polynomial background subtraction (order 4–6) and local-maxima peak
  picking, then intensity normalization.
* Primitive cells as the structure the flow model was trained on.

Released CNRS/RRUFF CSVs are not a single native axis: some are already Cu 2θ
in degrees, some are radians, some are still synchrotron 2θ. Axis and
wavelength are inferred by matching the paired CIF, not by trusting opXRD
metadata (which is often the original instrument λ of a pattern that was
later rewritten).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from scipy.signal import find_peaks

CU_KA = 1.5406
TRAIN_TMIN = 5.0
TRAIN_TMAX = 80.0
GRID_STEP = 0.02
MAX_PEAKS = 179
POLY_ORDER = 6
SMOOTH_WINDOW = 3
PEAK_HEIGHT = 5.0  # matches training mask pxrd_y > 5 after scaling to 100
PEAK_PROMINENCE = 0.05

CANDIDATE_WAVELENGTHS = (
    1.5406,
    1.54184,
    1.7902,   # Co
    0.71073,  # Mo
    1.9373,   # Fe
    2.2897,   # Cr
    0.5594,   # Ag
)


@dataclass
class AxisGuess:
    kind: str  # two_theta_deg | two_theta_rad | q | d
    wavelength: float | None
    score: float
    n_peaks: int


def d_from_two_theta(two_theta_deg: float, wavelength: float) -> float | None:
    if two_theta_deg <= 0 or wavelength <= 0:
        return None
    s = math.sin(math.radians(two_theta_deg) / 2.0)
    if s <= 0:
        return None
    return wavelength / (2.0 * s)


def two_theta_from_d(d: float, wavelength: float) -> float | None:
    if d <= 0 or wavelength <= 0:
        return None
    arg = wavelength / (2.0 * d)
    if not (0.0 < arg < 1.0):
        return None
    return math.degrees(2.0 * math.asin(arg))


def d_from_q(q: float) -> float | None:
    if q <= 0:
        return None
    return 2.0 * math.pi / q


def native_to_d(x: np.ndarray, kind: str, wavelength: float | None) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for i, xi in enumerate(x):
        if kind == "two_theta_deg":
            d = d_from_two_theta(float(xi), float(wavelength or CU_KA))
        elif kind == "two_theta_rad":
            d = d_from_two_theta(math.degrees(float(xi)), float(wavelength or CU_KA))
        elif kind == "q":
            d = d_from_q(float(xi))
        elif kind == "d":
            d = float(xi) if xi > 0 else None
        else:
            raise ValueError(kind)
        if d is not None:
            out[i] = d
    return out


def d_to_cu_two_theta(d: np.ndarray) -> np.ndarray:
    tt = np.full(d.shape, np.nan, dtype=np.float64)
    for i, di in enumerate(d):
        if not np.isfinite(di):
            continue
        t = two_theta_from_d(float(di), CU_KA)
        if t is not None:
            tt[i] = t
    return tt


def remap_profile_to_cu_ka(
    x: np.ndarray, y: np.ndarray, kind: str, wavelength: float | None
) -> tuple[np.ndarray, np.ndarray]:
    d = native_to_d(np.asarray(x, dtype=np.float64), kind, wavelength)
    tt = d_to_cu_two_theta(d)
    yy = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(tt) & np.isfinite(yy)
    tt, yy = tt[keep], yy[keep]
    if tt.size == 0:
        return tt, yy
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    _, uniq = np.unique(np.round(tt, 6), return_index=True)
    return tt[uniq], yy[uniq]


def interpolate_train_grid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 4:
        return x, y
    lo = max(TRAIN_TMIN, float(np.min(x)))
    hi = min(TRAIN_TMAX, float(np.max(x)))
    if hi - lo < 1.0:
        keep = (x >= lo) & (x <= hi)
        return x[keep], y[keep]
    grid = np.arange(lo, hi + 0.5 * GRID_STEP, GRID_STEP)
    return grid, np.interp(grid, x, y)


def subtract_polynomial_background(x: np.ndarray, y: np.ndarray, poly_order: int = POLY_ORDER) -> np.ndarray:
    if y.size < poly_order + 2:
        return np.maximum(y, 0.0)
    window = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 else SMOOTH_WINDOW + 1
    smoothed = np.convolve(y, np.ones(window) / window, mode="same")
    background = Polynomial.fit(x, smoothed, min(poly_order, max(1, y.size - 2)))(x)
    corrected = np.maximum(0.0, smoothed - background)
    vmax = float(np.max(corrected)) if corrected.size else 0.0
    if vmax <= 0:
        return corrected
    return corrected / vmax * 100.0


def search_peaks(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y.size == 0 or float(np.max(y)) <= 0:
        return x[:0], y[:0]
    prominence = (float(np.max(y)) - float(np.min(y))) * PEAK_PROMINENCE
    idx, _ = find_peaks(y, height=PEAK_HEIGHT, prominence=max(prominence, 1e-6))
    px, py = x[idx], y[idx]
    if py.size == 0:
        return px, py
    py = py / float(np.max(py)) * 100.0
    keep = (px >= TRAIN_TMIN) & (px <= TRAIN_TMAX) & (px < 180.0)
    px, py = px[keep], py[keep]
    if px.size > MAX_PEAKS:
        order = np.sort(np.argsort(py)[::-1][:MAX_PEAKS])
        px, py = px[order], py[order]
    return px, py


def peaks_from_cu_profile(x_cu: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xg, yg = interpolate_train_grid(x_cu, y)
    y_corr = subtract_polynomial_background(xg, yg)
    return search_peaks(xg, y_corr)


def simulated_d_list(structure: Structure) -> tuple[np.ndarray, np.ndarray]:
    pat = XRDCalculator(wavelength=CU_KA).get_pattern(
        structure, scaled=True, two_theta_range=(TRAIN_TMIN, TRAIN_TMAX)
    )
    ds = []
    for tt in pat.x:
        d = d_from_two_theta(float(tt), CU_KA)
        ds.append(d if d is not None else np.nan)
    return np.asarray(ds, dtype=np.float64), np.asarray(pat.y, dtype=np.float64)


def peak_match_score(obs_two_theta: np.ndarray, sim_d: np.ndarray, sim_y: np.ndarray) -> float:
    if obs_two_theta.size == 0 or sim_d.size == 0:
        return 0.0
    obs_d = []
    for tt in obs_two_theta:
        d = d_from_two_theta(float(tt), CU_KA)
        if d is not None:
            obs_d.append(d)
    if not obs_d:
        return 0.0
    order = np.argsort(sim_y)[::-1][:40]
    sd = sim_d[order]
    sd = sd[np.isfinite(sd)]
    if sd.size == 0:
        return 0.0
    n = min(15, len(obs_d))
    hits = 0
    for d in obs_d[:n]:
        rel = float(np.min(np.abs(sd - d) / max(d, 1e-6)))
        if rel < 0.025:
            hits += 1
    return hits / n


def candidate_axes(x: np.ndarray, hinted_wavelength: float | None) -> list[tuple[str, float | None]]:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    out: list[tuple[str, float | None]] = [("two_theta_deg", CU_KA)]
    if hinted_wavelength is not None and abs(hinted_wavelength - CU_KA) > 0.01:
        out.append(("two_theta_deg", float(hinted_wavelength)))
    if xmax < 5.0:
        out.append(("two_theta_rad", CU_KA))
        if hinted_wavelength:
            out.append(("two_theta_rad", float(hinted_wavelength)))
    if xmax < 8.0:
        out.append(("q", None))
    if xmax < 20.0:
        out.append(("d", None))
    for lam in CANDIDATE_WAVELENGTHS:
        if abs(lam - CU_KA) < 1e-6:
            continue
        if xmin >= 3.0:  # looks like degrees already
            out.append(("two_theta_deg", lam))
    # de-duplicate
    seen: set[tuple[str, float | None]] = set()
    uniq = []
    for kind, lam in out:
        key = (kind, None if lam is None else round(float(lam), 5))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((kind, lam))
    return uniq


def infer_and_convert(
    x: np.ndarray,
    y: np.ndarray,
    structure: Structure,
    hinted_wavelength: float | None = None,
) -> dict[str, Any]:
    sim_d, sim_y = simulated_d_list(structure)
    ranked: list[tuple[float, AxisGuess, np.ndarray, np.ndarray]] = []
    for kind, lam in candidate_axes(x, hinted_wavelength):
        x_cu, y_cu = remap_profile_to_cu_ka(x, y, kind, lam)
        if x_cu.size < 8:
            continue
        coverage = float(x_cu.size) / float(max(len(x), 1))
        if coverage < 0.4:
            continue
        px, py = peaks_from_cu_profile(x_cu, y_cu)
        match = peak_match_score(px, sim_d, sim_y)
        low_angle = 1.0 if (x_cu.size and float(np.min(x_cu)) < 25.0) else 0.5
        score = match * coverage * low_angle
        guess = AxisGuess(kind=kind, wavelength=lam, score=score, n_peaks=int(px.size))
        ranked.append((score, guess, x_cu, y_cu))
    ranked.sort(
        key=lambda t: (
            -t[0],
            0 if t[1].kind == "two_theta_deg" and t[1].wavelength == CU_KA else 1,
        )
    )
    if not ranked:
        x_cu, y_cu = remap_profile_to_cu_ka(x, y, "two_theta_deg", CU_KA)
        px, py = peaks_from_cu_profile(x_cu, y_cu)
        return {
            "kind": "two_theta_deg",
            "wavelength": CU_KA,
            "score": 0.0,
            "pxrd_x": px,
            "pxrd_y": py,
            "x_cu": x_cu,
            "y_cu": y_cu,
            "n_profile_cu": int(x_cu.size),
        }
    score, guess, x_cu, y_cu = ranked[0]
    px, py = peaks_from_cu_profile(x_cu, y_cu)
    return {
        "kind": guess.kind,
        "wavelength": guess.wavelength,
        "score": float(score),
        "n_peaks": int(px.size),
        "pxrd_x": px,
        "pxrd_y": py,
        "x_cu": x_cu,
        "y_cu": y_cu,
        "n_profile_cu": int(x_cu.size),
    }


def to_primitive(structure: Structure) -> Structure:
    try:
        prim = structure.get_primitive_structure()
    except Exception:
        return structure
    if len(prim) == 0:
        return structure
    return prim


def opxrd_hint(cif_path: Path, opxrd_dir: Path | None) -> float | None:
    if opxrd_dir is None or not opxrd_dir.exists():
        return None
    first = cif_path.open().readline().strip()
    match = re.search(r"pattern_(\d+)", first)
    if not match:
        return None
    blob_path = opxrd_dir / f"pattern_{match.group(1)}.json"
    if not blob_path.exists():
        return None
    blob = json.loads(blob_path.read_text())
    label = blob.get("label")
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            return None
    if not isinstance(label, dict):
        return None
    info = label.get("xray_info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    if not isinstance(info, dict):
        return None
    raw = info.get("primary_wavelength") or info.get("wavelength")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if 0.2 < value < 3.0 else None
