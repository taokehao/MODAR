"""Physical-consistency reranking of generated crystal candidates.

Backbone-agnostic v1 reranker:

    R_pxrd      re-simulated PXRD vs the condition peaks (rasterized cosine)
    R_nd        re-simulated neutron pattern vs condition peaks (if provided)
    R_charge    SMACT charge balance of the composition
    R_validity  minimum interatomic distance / volume-per-atom sanity gate
    dedup       near-duplicate candidates collapsed before ranking

Only modalities present in the condition prompt are scored, so the reranker
never uses information the model itself did not receive.
"""

from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

GRID_MIN = 5.0
GRID_MAX = 80.0
GRID_STEP = 0.05
SIGMA = 0.25
MAX_PEAKS = 24

W_PXRD = 1.0
# Tuned via rerank_ablation.py on the Tier-A ladder: 0.25 edges out 0.5 in
# lattice-free (and never hurts known-lattice); 0.1 starts dropping useful signal.
W_ND = 0.25
W_CHARGE = 0.2
INVALID_PENALTY = -2.0
# Noisy-aware match geometry (noisy_rerank_ablation.py spectrum mode, §8.11.2):
# under experimental artifacts, broader raster σ + mild 2θ shift search lifts
# lattice-free Top-1 ~+3pp on the frozen noisy ladder without hurting T20.
NOISY_MATCH_SIGMA = 0.40
NOISY_SHIFT_MAX = 0.20
NOISY_SHIFT_STEP = 0.05

MIN_DIST = 0.5
MIN_VOL_PER_ATOM = 2.0
MAX_VOL_PER_ATOM = 150.0

_NUM_CHARS = set("0123456789.-")


# ---------------------------------------------------------------------------
# Reference peaks from condition prompts
# ---------------------------------------------------------------------------

def peaks_from_condition_tokens(
    tokens: list[str], start_marker: str, end_marker: str
) -> list[tuple[float, float]] | None:
    """Parse (two_theta, intensity) pairs from a <PEAK> t .. i .. w .. block."""
    if start_marker not in tokens:
        return None
    i = tokens.index(start_marker) + 1
    peaks: list[tuple[float, float]] = []

    def read_number(j: int) -> tuple[float, int]:
        chars: list[str] = []
        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j] in _NUM_CHARS:
            chars.append(tokens[j])
            j += 1
        return float("".join(chars)), j

    while i < len(tokens) and tokens[i] == "<PEAK>":
        i += 1
        values: dict[str, float] = {}
        for key in ("t", "i", "w"):
            if i >= len(tokens) or tokens[i] != key:
                return peaks or None
            i += 1
            try:
                values[key], i = read_number(i)
            except ValueError:
                return peaks or None
        peaks.append((values["t"], values["i"]))
        if i < len(tokens) and tokens[i] == end_marker:
            break
    return peaks or None


def rasterize_peaks(
    peaks: list[tuple[float, float]],
    grid_min: float = GRID_MIN,
    grid_max: float = GRID_MAX,
    step: float = GRID_STEP,
    sigma: float = SIGMA,
) -> np.ndarray:
    grid = np.arange(grid_min, grid_max + step, step)
    vec = np.zeros_like(grid)
    inv = 1.0 / (2.0 * sigma * sigma)
    for pos, intensity in peaks:
        if intensity <= 0:
            continue
        vec += intensity * np.exp(-((grid - pos) ** 2) * inv)
    return vec


def spectrum_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _shift_peaks(
    peaks: list[tuple[float, float]], shift: float
) -> list[tuple[float, float]]:
    return [(t + shift, i) for t, i in peaks]


def best_spectrum_cosine(
    cand_peaks: list[tuple[float, float]],
    ref_peaks: list[tuple[float, float]],
    *,
    sigma: float,
    shift_max: float = 0.0,
    shift_step: float = NOISY_SHIFT_STEP,
) -> float:
    """Cosine between candidate and reference peaks, optionally max over 2θ shifts."""
    cand = rasterize_peaks(cand_peaks, sigma=sigma)
    if shift_max <= 0:
        return spectrum_cosine(cand, rasterize_peaks(ref_peaks, sigma=sigma))
    best = 0.0
    n_step = int(round(shift_max / max(shift_step, 1e-6)))
    for k in range(-n_step, n_step + 1):
        shifted = _shift_peaks(ref_peaks, k * shift_step)
        best = max(best, spectrum_cosine(cand, rasterize_peaks(shifted, sigma=sigma)))
    return best


# ---------------------------------------------------------------------------
# Candidate-side simulation and scores (run in worker processes)
# ---------------------------------------------------------------------------

def _structure_from_payload(payload: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    lat = payload["lattice"]
    lattice = Lattice.from_parameters(
        lat["a"], lat["b"], lat["c"], lat["alpha"], lat["beta"], lat["gamma"]
    )
    return Structure(lattice, payload["species"], payload["frac_coords"])


def _top_peaks(x: np.ndarray, y: np.ndarray, max_peaks: int = MAX_PEAKS) -> list[tuple[float, float]]:
    if len(x) == 0:
        return []
    ymax = float(np.max(y))
    if ymax <= 0:
        return []
    order = np.argsort(-y)[:max_peaks]
    return [(float(x[j]), float(y[j] / ymax)) for j in sorted(order, key=lambda j: x[j])]


def _simulate_scores(args: tuple) -> dict[str, float]:
    """Worker: simulate PXRD/ND for one candidate structure and score validity.

    ``args`` is ``(payload, want_pxrd, want_nd, match_sigma, shift_max, shift_step)``.
    Reference peaks (not pre-rasterized vectors) live on the payload under
    ``_ref_pxrd_peaks`` / ``_ref_nd_peaks`` so shift search can move them.
    """
    payload, want_pxrd, want_nd, match_sigma, shift_max, shift_step = args
    out: dict[str, float] = {
        "valid": 0.0,
        "pxrd_cos": float("nan"),
        "nd_cos": float("nan"),
        "sim_failed": 0.0,
    }
    try:
        structure = _structure_from_payload(payload)
    except Exception:
        out["sim_failed"] = 1.0
        return out

    try:
        n = len(structure)
        vol_per_atom = structure.volume / max(1, n)
        min_dist = float("inf")
        if n > 1:
            dm = structure.distance_matrix
            min_dist = float(np.min(dm[np.triu_indices(n, k=1)]))
        valid = (
            MIN_VOL_PER_ATOM <= vol_per_atom <= MAX_VOL_PER_ATOM
            and (n == 1 or min_dist >= MIN_DIST)
        )
        out["valid"] = float(valid)
    except Exception:
        out["sim_failed"] = 1.0
        return out

    if not out["valid"]:
        # Degenerate cells make the diffraction calculators slow/fragile.
        return out

    if want_pxrd:
        try:
            from pymatgen.analysis.diffraction.xrd import XRDCalculator

            pattern = XRDCalculator().get_pattern(
                structure, scaled=True, two_theta_range=(GRID_MIN, GRID_MAX)
            )
            peaks = _top_peaks(np.asarray(pattern.x), np.asarray(pattern.y))
            ref = payload.get("_ref_pxrd_peaks")
            if ref:
                out["pxrd_cos"] = best_spectrum_cosine(
                    peaks,
                    ref,
                    sigma=match_sigma,
                    shift_max=shift_max,
                    shift_step=shift_step,
                )
        except Exception:
            out["sim_failed"] = 1.0
    if want_nd:
        try:
            from pymatgen.analysis.diffraction.neutron import NDCalculator

            pattern = NDCalculator(wavelength=1.54184).get_pattern(
                structure, scaled=True, two_theta_range=(GRID_MIN, GRID_MAX)
            )
            peaks = _top_peaks(np.asarray(pattern.x), np.asarray(pattern.y))
            ref = payload.get("_ref_nd_peaks")
            if ref:
                out["nd_cos"] = best_spectrum_cosine(
                    peaks,
                    ref,
                    sigma=match_sigma,
                    shift_max=shift_max,
                    shift_step=shift_step,
                )
        except Exception:
            # ND tables miss some isotopes; not a hard failure.
            pass
    return out


# ---------------------------------------------------------------------------
# Charge balance (composition level, cached)
# ---------------------------------------------------------------------------

_charge_cache: dict[tuple[tuple[str, int], ...], bool] = {}


def smact_charge_balanced(composition: list[tuple[str, int]]) -> bool:
    key = tuple(sorted((str(el), int(n)) for el, n in composition))
    cached = _charge_cache.get(key)
    if cached is not None:
        return cached
    try:
        import smact
        from smact import screening

        elements = [el for el, _ in key]
        counts = [n for _, n in key]
        if len(elements) == 1:
            result = True
        else:
            space = {el: smact.Element(el) for el in elements}
            ox_lists = [space[el].oxidation_states for el in elements]
            result = False
            for combo in _ox_combinations(ox_lists):
                ok, _ = screening.neutral_ratios(combo, stoichs=[[c] for c in counts])
                if ok:
                    result = True
                    break
    except Exception:
        result = True  # never punish candidates when smact itself fails
    _charge_cache[key] = result
    return result


def _ox_combinations(ox_lists: list[list[int]], limit: int = 20000):
    import itertools

    total = 1
    for lst in ox_lists:
        total *= max(1, len(lst))
    if total > limit:
        return
    yield from itertools.product(*ox_lists)


# ---------------------------------------------------------------------------
# Dedup + main entry point
# ---------------------------------------------------------------------------

def structure_dedup_key(payload: dict[str, Any]) -> tuple:
    lat = payload["lattice"]
    lat_key = tuple(round(float(lat[k]), 1) for k in ("a", "b", "c", "alpha", "beta", "gamma"))
    site_key = tuple(
        sorted(
            (sp, round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2))
            for sp, c in zip(payload["species"], payload["frac_coords"])
        )
    )
    return lat_key, site_key


@dataclass
class RerankResult:
    order: list[int]           # candidate indices, best first (deduped reps first)
    scores: list[float | None]  # per-candidate combined score (None: not scored)
    detail: list[dict[str, Any]]


def rerank_candidates(
    payloads: list[dict[str, Any] | None],
    ref_pxrd: list[tuple[float, float]] | None,
    ref_nd: list[tuple[float, float]] | None,
    executor: ProcessPoolExecutor | None = None,
    *,
    match_sigma: float = SIGMA,
    shift_max: float = 0.0,
    shift_step: float = NOISY_SHIFT_STEP,
) -> RerankResult:
    """Rank candidates by physical-consistency score.

    ``payloads[i]`` is None for parse-invalid candidates (they sink to the end,
    keeping their relative order). Duplicate structures are scored once; only
    the first member of each duplicate group is kept in the top section of the
    returned order, the rest follow after all unique candidates.

    ``match_sigma`` / ``shift_max`` control how reference peaks are compared to
    re-simulated candidate patterns. Clean evals keep the defaults; noisy evals
    should pass ``NOISY_MATCH_SIGMA`` / ``NOISY_SHIFT_MAX`` (§8.11.2).
    """
    n = len(payloads)
    scores: list[float | None] = [None] * n
    detail: list[dict[str, Any]] = [{} for _ in range(n)]

    groups: dict[tuple, list[int]] = {}
    for i, payload in enumerate(payloads):
        if payload is None:
            continue
        groups.setdefault(structure_dedup_key(payload), []).append(i)

    reps = [members[0] for members in groups.values()]
    jobs = []
    for i in reps:
        payload = dict(payloads[i])
        payload["_ref_pxrd_peaks"] = ref_pxrd
        payload["_ref_nd_peaks"] = ref_nd
        jobs.append(
            (
                payload,
                ref_pxrd is not None,
                ref_nd is not None,
                float(match_sigma),
                float(shift_max),
                float(shift_step),
            )
        )

    if executor is not None and len(jobs) > 1:
        results = list(executor.map(_simulate_scores, jobs, chunksize=4))
    else:
        results = [_simulate_scores(job) for job in jobs]

    for i, res in zip(reps, results):
        payload = payloads[i]
        comp_counts = sorted(Counter(payload["species"]).items())
        charge_ok = smact_charge_balanced(comp_counts)

        score = 0.0
        if not math.isnan(res["pxrd_cos"]):
            score += W_PXRD * res["pxrd_cos"]
        if not math.isnan(res["nd_cos"]):
            score += W_ND * res["nd_cos"]
        score += W_CHARGE * float(charge_ok)
        if res["valid"] < 0.5 or res["sim_failed"] > 0.5:
            score += INVALID_PENALTY
        det = {
            "pxrd_cos": None if math.isnan(res["pxrd_cos"]) else round(res["pxrd_cos"], 4),
            "nd_cos": None if math.isnan(res["nd_cos"]) else round(res["nd_cos"], 4),
            "valid": bool(res["valid"]),
            "charge_ok": bool(charge_ok),
            "sim_failed": bool(res["sim_failed"]),
            "match_sigma": float(match_sigma),
            "shift_max": float(shift_max),
        }
        for member in groups[structure_dedup_key(payload)]:
            scores[member] = score
            detail[member] = dict(det)

    unique_section = sorted(reps, key=lambda i: (-(scores[i] or float("-inf")), i))
    dup_section = [
        member
        for key, members in groups.items()
        for member in members[1:]
    ]
    dup_section.sort(key=lambda i: (-(scores[i] or float("-inf")), i))
    invalid_section = [i for i in range(n) if payloads[i] is None]
    order = unique_section + dup_section + invalid_section
    return RerankResult(order=order, scores=scores, detail=detail)


__all__ = [
    "NOISY_MATCH_SIGMA",
    "NOISY_SHIFT_MAX",
    "NOISY_SHIFT_STEP",
    "RerankResult",
    "best_spectrum_cosine",
    "peaks_from_condition_tokens",
    "rasterize_peaks",
    "rerank_candidates",
    "smact_charge_balanced",
    "spectrum_cosine",
    "structure_dedup_key",
]
