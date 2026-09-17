"""Deterministic peak extraction for PXRD / neutron 2-theta profiles.

The exported multimodal records store full diffraction profiles sampled on a
2-theta grid with intensities normalized to a maximum of 100. For the first
language-model version, profiles are reduced to a compact set of peak tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Peak:
    two_theta: float
    intensity: float
    width: float


def _local_maxima(y: list[float]) -> list[int]:
    idx: list[int] = []
    n = len(y)
    for i in range(n):
        left = y[i - 1] if i > 0 else float("-inf")
        right = y[i + 1] if i < n - 1 else float("-inf")
        if y[i] > left and y[i] >= right:
            idx.append(i)
    return idx


def _estimate_width(x: list[float], y: list[float], peak_idx: int) -> float:
    half = y[peak_idx] / 2.0
    n = len(y)
    left = peak_idx
    while left > 0 and y[left] > half:
        left -= 1
    right = peak_idx
    while right < n - 1 and y[right] > half:
        right += 1
    width = abs(x[right] - x[left])
    if width <= 0:
        step = (x[-1] - x[0]) / max(1, n - 1)
        width = step
    return width


def extract_peaks(
    x: list[float] | None,
    y: list[float] | None,
    max_peaks: int = 24,
    rel_threshold: float = 0.02,
) -> list[Peak]:
    if not x or not y or len(x) != len(y):
        return []
    peak_max = max(y)
    if peak_max <= 0:
        return []
    threshold = rel_threshold * peak_max
    candidates: list[Peak] = []
    for i in _local_maxima(y):
        if y[i] < threshold:
            continue
        candidates.append(
            Peak(
                two_theta=float(x[i]),
                intensity=float(y[i] / peak_max),
                width=float(_estimate_width(x, y, i)),
            )
        )
    candidates.sort(key=lambda p: p.intensity, reverse=True)
    selected = candidates[:max_peaks]
    selected.sort(key=lambda p: p.two_theta)
    return selected
