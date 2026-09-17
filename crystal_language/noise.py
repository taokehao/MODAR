"""Experimental-noise augmentation for multimodal crystal conditions.

The model stores PXRD / neutron as dense spectra and XRF / XPS as structured
condition tokens.  This module perturbs both representations at load/eval time
without changing the target crystal sequence.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch

from .peak_encoder import peaks_to_di
from .spectra_encoder import rasterize_spectrum


NOISE_NONE = "none"
NOISE_EXPERIMENTAL = "experimental"


def _num_tokens(value: float, decimals: int) -> list[str]:
    text = f"{float(value):.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return list(text)


def _read_number(tokens: list[str], start: int) -> tuple[float | None, int]:
    chars: list[str] = []
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("<") or tok.isalpha():
            break
        chars.append(tok)
        i += 1
    if not chars:
        return None, i
    try:
        return float("".join(chars)), i
    except ValueError:
        return None, i


def perturb_condition_block(
    name: str,
    block_tokens: list[str],
    rng: random.Random,
    profile: str = NOISE_NONE,
) -> list[str]:
    """Perturb structured token conditions (currently XRF and XPS-valence)."""
    if profile == NOISE_NONE:
        return block_tokens
    if profile != NOISE_EXPERIMENTAL:
        raise ValueError(f"unknown noise profile: {profile!r}")
    if name == "xrf":
        return _perturb_xrf(block_tokens, rng)
    if name == "xps":
        return _perturb_xps(block_tokens, rng)
    return block_tokens


def _perturb_xrf(tokens: list[str], rng: random.Random) -> list[str]:
    # Quantitative XRF uncertainty is commonly ~5-10% relative for minor
    # elements / surface-geometry effects.  Apply log-normal relative noise and
    # renormalize mass fractions to keep the composition block physically sane.
    entries: list[tuple[str, float]] = []
    i = 0
    while i < len(tokens):
        if tokens[i] != "<XRF_MASS_FRAC>" or i + 1 >= len(tokens):
            i += 1
            continue
        element = tokens[i + 1]
        value, j = _read_number(tokens, i + 2)
        if value is not None:
            noisy = max(0.0, value * math.exp(rng.gauss(0.0, 0.08)))
            entries.append((element, noisy))
        i = max(j, i + 2)
    total = sum(v for _, v in entries)
    if not entries or total <= 0:
        return tokens
    out = ["<XRF_START>"]
    for element, value in entries:
        out.extend(["<XRF_MASS_FRAC>", element])
        out.extend(_num_tokens(value / total, 3))
    out.append("<XRF_END>")
    return out


def _perturb_xps(tokens: list[str], rng: random.Random) -> list[str]:
    # The current representation is not a full XP spectrum; it is a per-element
    # valence proxy.  Map experimental peak-fit / binding-energy ambiguity to a
    # small oxidation-state uncertainty while preserving the existing grammar.
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        if tok == "<VALENCE>" and i + 1 < len(tokens):
            element = tokens[i + 1]
            out.append(element)
            value, j = _read_number(tokens, i + 2)
            if value is not None:
                noisy = max(-4.0, min(8.0, value + rng.gauss(0.0, 0.2)))
                out.extend(_num_tokens(noisy, 1))
                i = j
                continue
            i += 2
            continue
        i += 1
    return out


def rasterize_noisy_spectrum(
    x: list[float],
    y: list[float],
    *,
    name: str,
    rng: random.Random,
    profile: str = NOISE_NONE,
    grid_min: float = 5.0,
    grid_max: float = 80.0,
    step: float = 0.05,
    sigma: float = 0.15,
) -> torch.Tensor:
    """Rasterize a spectrum and optionally add experimental artifacts."""
    if profile == NOISE_NONE:
        return rasterize_spectrum(x, y, grid_min=grid_min, grid_max=grid_max, step=step, sigma=sigma)
    if profile != NOISE_EXPERIMENTAL:
        raise ValueError(f"unknown noise profile: {profile!r}")
    if not x or not y:
        return rasterize_spectrum([], [], grid_min=grid_min, grid_max=grid_max, step=step, sigma=sigma)

    # Powder-diffraction augmentation literature commonly varies peak shifts,
    # broadening, texture/intensity, polynomial background and Gaussian/Poisson
    # counting noise.  Use conservative online ranges so the model sees realistic
    # lab imperfections without destroying peak identity.
    if name == "pxrd":
        global_shift = rng.uniform(-0.20, 0.20)  # zero/sample-height error (deg 2theta)
        jitter_std = 0.015
        broaden = rng.uniform(0.10, 0.35)
        bg_max = rng.uniform(0.00, 0.08)
        noise_rel = rng.uniform(0.003, 0.05)
        texture_sigma = 0.25
    else:  # neutron powder diffraction: similar profile, slightly less zero shift.
        global_shift = rng.uniform(-0.12, 0.12)
        jitter_std = 0.012
        broaden = rng.uniform(0.10, 0.30)
        bg_max = rng.uniform(0.00, 0.06)
        noise_rel = rng.uniform(0.003, 0.04)
        texture_sigma = 0.20

    xp: list[float] = []
    yp: list[float] = []
    for xi, yi in zip(x, y):
        xp.append(float(xi) + global_shift + rng.gauss(0.0, jitter_std))
        yp.append(max(0.0, float(yi) * math.exp(rng.gauss(0.0, texture_sigma))))
    spec = rasterize_spectrum(
        xp,
        yp,
        grid_min=grid_min,
        grid_max=grid_max,
        step=step,
        sigma=max(sigma, broaden),
    )
    if spec.numel() == 0:
        return spec

    grid = torch.linspace(0.0, 1.0, spec.numel(), dtype=spec.dtype)
    c0 = rng.uniform(0.0, bg_max)
    c1 = rng.uniform(-0.5 * bg_max, 0.5 * bg_max)
    c2 = rng.uniform(-0.5 * bg_max, 0.5 * bg_max)
    background = c0 + c1 * grid + c2 * (grid - 0.5) ** 2
    noisy = spec + background.clamp_min(0.0)
    noisy = noisy + torch.randn_like(noisy) * float(noise_rel)
    noisy = noisy.clamp_min(0.0)
    maxv = float(noisy.max())
    if maxv > 0:
        noisy = noisy / maxv
    return noisy


def noisy_peaks_to_di(
    x: list[float],
    y: list[float],
    *,
    name: str,
    rng: random.Random,
    wavelength: float,
    max_peaks: int = 128,
    profile: str = NOISE_NONE,
) -> torch.Tensor:
    """Build a (d, I) peak tensor, optionally with peak-list artifacts.

    The dense path perturbs a raster; here the perturbations act on the peak
    list itself, which is where real peak-picking errors live: weak reflections
    fall below the detection threshold, close reflections merge, positions carry
    a zero-point error, and impurity phases add reflections that belong to no
    phase in the target. Angular range and step-size effects are applied in
    2-theta before the conversion to d, because that is where they physically
    act.
    """
    if profile == NOISE_NONE:
        return peaks_to_di(x, y, wavelength=wavelength, max_peaks=max_peaks)
    if profile != NOISE_EXPERIMENTAL:
        raise ValueError(f"unknown noise profile: {profile!r}")
    if not x or not y:
        return peaks_to_di([], [], wavelength=wavelength, max_peaks=max_peaks)

    if name == "pxrd":
        shift_max, jitter_std, texture_sigma = 0.20, 0.02, 0.25
        drop_frac = rng.uniform(0.0, 0.40)
        n_fake = rng.randint(0, 5)
    else:
        shift_max, jitter_std, texture_sigma = 0.12, 0.015, 0.20
        drop_frac = rng.uniform(0.0, 0.30)
        n_fake = rng.randint(0, 3)

    global_shift = rng.uniform(-shift_max, shift_max)
    merge_eps = rng.choice([0.0, 0.05, 0.10, 0.20])
    step = rng.choice([0.0, 0.01, 0.02, 0.05, 0.10])
    lo = rng.uniform(5.0, 15.0) if rng.random() < 0.4 else 5.0
    hi = rng.uniform(55.0, 80.0) if rng.random() < 0.4 else 80.0

    pairs: list[tuple[float, float]] = []
    for xi, yi in zip(x, y):
        inten = float(yi)
        if inten <= 0.0:
            continue
        pos = float(xi) + global_shift + rng.gauss(0.0, jitter_std)
        if pos < lo or pos > hi:
            continue
        if step > 0.0:
            pos = round(pos / step) * step
        pairs.append((pos, inten * math.exp(rng.gauss(0.0, texture_sigma))))
    if not pairs:
        return peaks_to_di([], [], wavelength=wavelength, max_peaks=max_peaks)

    ymax = max(p[1] for p in pairs)
    weak = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    n_drop = int(round(drop_frac * len(pairs)))
    dropped = set(weak[:n_drop])
    # Never drop the strongest reflection: a peak list without its own maximum
    # is not a peak-picking failure mode, it is a different pattern.
    strongest = max(range(len(pairs)), key=lambda i: pairs[i][1])
    dropped.discard(strongest)
    pairs = [p for i, p in enumerate(pairs) if i not in dropped]

    if merge_eps > 0.0 and len(pairs) > 1:
        pairs.sort(key=lambda p: p[0])
        merged: list[list[float]] = [list(pairs[0])]
        for pos, inten in pairs[1:]:
            if pos - merged[-1][0] <= merge_eps:
                total = merged[-1][1] + inten
                if total > 0:
                    merged[-1][0] = (
                        merged[-1][0] * merged[-1][1] + pos * inten
                    ) / total
                merged[-1][1] = total
            else:
                merged.append([pos, inten])
        pairs = [(p[0], p[1]) for p in merged]

    for _ in range(n_fake):
        pairs.append((rng.uniform(lo, hi), ymax * rng.uniform(0.02, 0.25)))

    return peaks_to_di(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        wavelength=wavelength,
        max_peaks=max_peaks,
    )


__all__ = [
    "NOISE_EXPERIMENTAL",
    "NOISE_NONE",
    "noisy_peaks_to_di",
    "perturb_condition_block",
    "rasterize_noisy_spectrum",
]
