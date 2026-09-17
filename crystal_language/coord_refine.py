"""Differentiable PXRD forward model and fractional-coordinate refinement.

Motivation: on the RealPXRD-Solver benchmarks the flow model is handed the
ground-truth cell and spends 200 continuous steps moving atoms under it. Our
decoder emits coordinates as discrete tokens in one pass and never revisits
them, so near-miss solutions stay near-misses. This module adds the missing
step -- with the cell and composition frozen, push fractional coordinates
downhill against the measured pattern.

Freezing the cell is what makes it cheap: every reflection's 2-theta is then
fixed, so the hkl list, scattering factors and Lorentz-polarisation weights are
precomputed once and only the structure factor depends on the coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
from pymatgen.core import Element, Lattice, Structure

# pymatgen's XRDCalculator constant for the s^2 term of the scattering factor.
SCATTERING_CONST = 41.78214
CU_KA = 1.54184


@dataclass
class RefineConfig:
    two_theta_min: float = 5.0
    two_theta_max: float = 80.0
    wavelength: float = CU_KA
    grid_step: float = 0.05
    sigma: float = 0.20
    steps: int = 400
    lr: float = 0.01
    # A powder pattern fixes only |F_hkl|, so the coordinate problem is
    # under-determined: unconstrained descent happily walks to a homometric
    # solution with cosine 1.0 that is a different crystal. The anchor keeps the
    # search inside a trust region around the decoder's proposal, turning this
    # into polishing rather than re-solving.
    w_anchor: float = 5.0
    anchor_radius: float = 0.08
    device: str = "cuda"


def _hkl_table(lattice: Lattice, cfg: RefineConfig) -> np.ndarray:
    """All hkl whose d-spacing falls inside the 2-theta window."""
    max_sin = np.sin(np.radians(cfg.two_theta_max / 2.0))
    min_sin = np.sin(np.radians(cfg.two_theta_min / 2.0))
    d_min = cfg.wavelength / (2.0 * max_sin)
    d_max = cfg.wavelength / (2.0 * min_sin) if min_sin > 0 else 1e6
    recip = lattice.reciprocal_lattice_crystallographic
    # Reciprocal-space sphere of radius 1/d_min covers the whole 2-theta window.
    points = recip.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.0 / d_min)
    out = []
    for frac, dist, _, _ in points:
        if dist <= 1e-8:
            continue
        d = 1.0 / dist
        if d < d_min or d > d_max:
            continue
        out.append((np.rint(frac).astype(int), d))
    return out


def _neutron_lengths(symbols: list[str]) -> np.ndarray:
    from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN

    return np.asarray([float(ATOMIC_SCATTERING_LEN[s]) for s in symbols], dtype=np.float64)


def _scattering_coeffs(symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    zs, coeffs = [], []
    for sym in symbols:
        el = Element(sym)
        zs.append(float(el.Z))
        c = ATOMIC_SCATTERING_PARAMS.get(sym)
        if c is None:
            c = [[0.0, 0.0]] * 4
        arr = np.asarray(c, dtype=np.float64)
        if arr.shape[0] < 4:
            arr = np.vstack([arr, np.zeros((4 - arr.shape[0], 2))])
        coeffs.append(arr[:4])
    return np.asarray(zs), np.asarray(coeffs)


class FixedLatticePXRD:
    """Precomputed PXRD operator for one frozen lattice and composition."""

    def __init__(self, lattice: Lattice, symbols: list[str], cfg: RefineConfig, neutron: bool = False):
        self.cfg = cfg
        self.neutron = neutron
        dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.device = dev

        table = _hkl_table(lattice, cfg)
        if not table:
            raise ValueError("no reflections in window")
        hkl = np.asarray([t[0] for t in table], dtype=np.float64)
        d = np.asarray([t[1] for t in table], dtype=np.float64)

        theta = np.arcsin(np.clip(cfg.wavelength / (2.0 * d), -1.0, 1.0))
        two_theta = np.degrees(2.0 * theta)
        s2 = (np.sin(theta) / cfg.wavelength) ** 2

        if neutron:
            # Neutron scattering lengths are Z-independent and may be negative,
            # so ND constrains |F| along a direction X-rays cannot see.
            f = np.repeat(_neutron_lengths(symbols)[:, None], s2.size, axis=1)
        else:
            zs, coeffs = _scattering_coeffs(symbols)
            f = zs[:, None] - SCATTERING_CONST * s2[None, :] * np.einsum(
                "ai,aih->ah",
                coeffs[:, :, 0],
                np.exp(-coeffs[:, :, 1][:, :, None] * s2[None, None, :]),
            )
        lp = (1.0 + np.cos(2.0 * theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))

        self.hkl = torch.tensor(hkl, dtype=torch.float32, device=dev)
        self.f = torch.tensor(f, dtype=torch.float32, device=dev)
        self.lp = torch.tensor(lp, dtype=torch.float32, device=dev)
        self.two_theta = torch.tensor(two_theta, dtype=torch.float32, device=dev)

        grid = np.arange(cfg.two_theta_min, cfg.two_theta_max + 1e-9, cfg.grid_step)
        self.grid = torch.tensor(grid, dtype=torch.float32, device=dev)
        # kernel[hkl, grid_bin] -- fixed because 2-theta positions never move.
        delta = self.two_theta[:, None] - self.grid[None, :]
        self.kernel = torch.exp(-0.5 * (delta / cfg.sigma) ** 2)

    def intensities(self, frac: torch.Tensor) -> torch.Tensor:
        """frac: (B, n_atoms, 3) -> (B, n_hkl) reflection intensities."""
        phase = 2.0 * torch.pi * torch.einsum("bad,hd->bah", frac, self.hkl)
        real = torch.einsum("bah,ah->bh", torch.cos(phase), self.f)
        imag = torch.einsum("bah,ah->bh", torch.sin(phase), self.f)
        return (real**2 + imag**2) * self.lp[None, :]

    def profile(self, frac: torch.Tensor) -> torch.Tensor:
        """frac: (B, n_atoms, 3) -> (B, n_grid) normalised profile."""
        inten = self.intensities(frac)
        prof = inten @ self.kernel
        return prof / prof.amax(dim=1, keepdim=True).clamp_min(1e-12)

    def observed_profile(self, peaks_x, peaks_y) -> torch.Tensor:
        x = torch.tensor(np.asarray(peaks_x, dtype=np.float64), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.asarray(peaks_y, dtype=np.float64), dtype=torch.float32, device=self.device)
        keep = (x >= self.cfg.two_theta_min) & (x <= self.cfg.two_theta_max)
        x, y = x[keep], y[keep]
        if x.numel() == 0:
            raise ValueError("no observed peaks in window")
        delta = x[:, None] - self.grid[None, :]
        prof = (y[:, None] * torch.exp(-0.5 * (delta / self.cfg.sigma) ** 2)).sum(dim=0)
        return prof / prof.amax().clamp_min(1e-12)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def refine_coords(
    structures: list[Structure],
    peaks_x,
    peaks_y,
    cfg: RefineConfig | None = None,
    nd_x=None,
    nd_y=None,
    w_nd: float = 0.0,
) -> tuple[list[Structure], np.ndarray, np.ndarray]:
    """Refine fractional coordinates of a batch sharing one lattice+composition.

    Pass ``nd_x``/``nd_y`` with ``w_nd > 0`` to add a neutron term. X-rays see
    the electron density and neutrons see the nuclei, so the two |F| constraints
    are largely independent and the joint objective is far less degenerate than
    PXRD alone.

    Returns the refined structures and the joint objective cosine after and
    before refinement.
    """
    cfg = cfg or RefineConfig()
    if not structures:
        return [], np.zeros(0), np.zeros(0)
    ref = structures[0]
    symbols = [s.specie.symbol for s in ref]
    op = FixedLatticePXRD(ref.lattice, symbols, cfg)
    obs = op.observed_profile(peaks_x, peaks_y)

    use_nd = w_nd > 0.0 and nd_x is not None and nd_y is not None and len(nd_x) > 0
    if use_nd:
        op_nd = FixedLatticePXRD(ref.lattice, symbols, cfg, neutron=True)
        obs_nd = op_nd.observed_profile(nd_x, nd_y)

    def objective(f: torch.Tensor) -> torch.Tensor:
        c = _cosine(op.profile(f), obs[None, :])
        if use_nd:
            c = (c + w_nd * _cosine(op_nd.profile(f), obs_nd[None, :])) / (1.0 + w_nd)
        return c

    frac0 = torch.tensor(
        np.stack([np.mod(s.frac_coords, 1.0) for s in structures]),
        dtype=torch.float32,
        device=op.device,
    )
    with torch.no_grad():
        cos_before = objective(frac0).cpu().numpy()

    frac = frac0.clone().requires_grad_(True)
    opt = torch.optim.Adam([frac], lr=cfg.lr)
    for _ in range(cfg.steps):
        opt.zero_grad()
        drift = torch.remainder(frac - frac0 + 0.5, 1.0) - 0.5
        excess = (drift.norm(dim=-1) - cfg.anchor_radius).clamp_min(0.0)
        loss = (1.0 - objective(frac) + cfg.w_anchor * (excess**2).mean(dim=-1)).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = torch.remainder(frac, 1.0)
        cos_after = objective(final).cpu().numpy()
        coords = final.cpu().numpy()

    out = []
    for i, s in enumerate(structures):
        out.append(Structure(s.lattice, [sp.specie for sp in s], coords[i]))
    return out, cos_after, cos_before
