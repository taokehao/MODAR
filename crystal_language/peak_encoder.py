"""Peak-set (d-I) diffraction encoders for the V2 crystal language model.

The dense CNN front-end in ``spectra_encoder`` consumes a fixed 2-theta raster,
so it inherits the measurement geometry: a pattern collected at a different
wavelength, angular range, or step size lands on a different grid and the
encoder sees a different object. Interplanar spacing removes that dependence,
because ``d`` is a property of the lattice rather than of the diffractometer.

This module encodes a variable-length list of ``(d, I)`` Bragg peaks with a
small permutation-equivariant transformer and reads out ``k_slots`` prefix
embeddings, matching the interface the decoder already expects at the
``<PXRD_EMB>`` / ``<ND_EMB>`` placeholder positions.

Padding convention: a peak row is padding when its intensity is <= 0, so the
dataset can emit a fixed ``[max_peaks, 2]`` tensor and the collate function
stays a plain ``torch.stack``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Cu-K-alpha 5-80 deg spans d = 1.20-17.7 A; log-d keeps that range well
# conditioned for the input MLPs.
D_MIN = 0.5
D_MAX = 40.0


def fourier_features(v: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """[..., 1] -> [..., 2 * n_freqs] sin/cos features at octave frequencies."""
    freqs = torch.arange(n_freqs, device=v.device, dtype=v.dtype)
    ang = v * (2.0**freqs) * math.pi
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class PeakSetEncoder(nn.Module):
    """Encode ``(d, I)`` peak lists into ``k_slots`` prefix embeddings."""

    def __init__(
        self,
        n_embd: int,
        k_slots: int = 16,
        n_layer: int = 3,
        n_head: int = 8,
        d_hidden: int = 256,
        n_freqs: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k_slots = k_slots
        self.n_freqs = n_freqs

        feat_dim = 1 + 2 * n_freqs
        self.d_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_embd),
        )
        self.i_mlp = nn.Sequential(
            nn.Linear(feat_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_embd),
        )
        self.slots = nn.Parameter(torch.zeros(k_slots, n_embd))
        nn.init.normal_(self.slots, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embd)

    def _featurize(self, peaks: torch.Tensor) -> torch.Tensor:
        d = peaks[..., 0:1].clamp(min=D_MIN, max=D_MAX)
        inten = peaks[..., 1:2].clamp(min=0.0, max=1.0)
        # log-d spreads the low-angle peaks that carry most of the cell
        # information; sqrt-I compresses the dynamic range of powder intensities.
        d_norm = torch.log(d / D_MIN) / math.log(D_MAX / D_MIN)
        i_norm = torch.sqrt(inten)
        d_feat = torch.cat([d_norm, fourier_features(d_norm, self.n_freqs)], dim=-1)
        i_feat = torch.cat([i_norm, fourier_features(i_norm, self.n_freqs)], dim=-1)
        return self.d_mlp(d_feat) + self.i_mlp(i_feat)

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        """peaks: [B, P, 2] of (d, intensity) -> [B, k_slots, n_embd]."""
        B = peaks.shape[0]
        pad = peaks[..., 1] <= 0.0
        h = self._featurize(peaks)
        slots = self.slots.to(h.dtype).unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([slots, h], dim=1)
        # Slot queries are never masked, so every position keeps at least the
        # slots as attendable keys and an all-padding pattern cannot produce NaN.
        key_padding = torch.cat(
            [torch.zeros(B, self.k_slots, dtype=torch.bool, device=peaks.device), pad],
            dim=1,
        )
        out = self.encoder(seq, src_key_padding_mask=key_padding)
        return self.norm(out[:, : self.k_slots, :])


class PeakSpectraEncoder(nn.Module):
    """Holds one peak-set encoder per supported modality."""

    MODALITIES = ("pxrd", "neutron")

    def __init__(
        self,
        n_embd: int,
        k_slots: int = 16,
        n_layer: int = 3,
        n_head: int = 8,
    ) -> None:
        super().__init__()
        self.k_slots = k_slots
        self.encoders = nn.ModuleDict(
            {
                name: PeakSetEncoder(
                    n_embd, k_slots=k_slots, n_layer=n_layer, n_head=n_head
                )
                for name in self.MODALITIES
            }
        )

    def forward(self, name: str, peaks: torch.Tensor) -> torch.Tensor:
        return self.encoders[name](peaks)


def two_theta_to_d(two_theta_deg: float, wavelength: float) -> float:
    """Bragg's law, guarding the small-angle limit."""
    theta = math.radians(0.5 * float(two_theta_deg))
    s = math.sin(theta)
    if s <= 1e-6:
        return D_MAX
    return float(wavelength) / (2.0 * s)


def peaks_to_di(
    xs: list[float],
    ys: list[float],
    wavelength: float,
    max_peaks: int = 128,
) -> torch.Tensor:
    """Convert a 2-theta peak list to a padded ``[max_peaks, 2]`` (d, I) tensor.

    Peaks are ranked by intensity so truncation drops the least informative
    reflections, then re-sorted by descending d to give the encoder a stable
    ordering (the transformer is permutation-equivariant, but a deterministic
    order keeps debugging and caching sane).
    """
    out = torch.zeros(max_peaks, 2, dtype=torch.float32)
    if not xs or not ys:
        return out
    pairs = [
        (two_theta_to_d(x, wavelength), float(y))
        for x, y in zip(xs, ys)
        if float(y) > 0.0
    ]
    if not pairs:
        return out
    ymax = max(p[1] for p in pairs)
    if ymax <= 0.0:
        return out
    pairs = [(d, i / ymax) for d, i in pairs]
    pairs.sort(key=lambda p: p[1], reverse=True)
    pairs = pairs[:max_peaks]
    pairs.sort(key=lambda p: p[0], reverse=True)
    for row, (d, i) in enumerate(pairs):
        out[row, 0] = d
        out[row, 1] = i
    return out


def d_to_two_theta(d: float, wavelength: float) -> float | None:
    """Inverse of :func:`two_theta_to_d`; ``None`` when no reflection exists."""
    if d <= 0.0:
        return None
    s = float(wavelength) / (2.0 * float(d))
    if s >= 1.0:
        return None
    return 2.0 * math.degrees(math.asin(s))


def peaks_from_di_tensor(
    peaks: "torch.Tensor", wavelength: float
) -> list[tuple[float, float]]:
    """Recover a ``(2-theta, intensity)`` peak list from a padded (d, I) tensor."""
    out: list[tuple[float, float]] = []
    for d, inten in peaks.tolist():
        if inten <= 0.0:
            continue
        tt = d_to_two_theta(d, wavelength)
        if tt is None:
            continue
        out.append((tt, float(inten)))
    out.sort(key=lambda p: p[0])
    return out


__all__ = [
    "PeakSetEncoder",
    "PeakSpectraEncoder",
    "d_to_two_theta",
    "fourier_features",
    "peaks_from_di_tensor",
    "peaks_to_di",
    "two_theta_to_d",
]
