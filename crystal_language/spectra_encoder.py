"""Dense diffraction-spectrum encoders for the V2 crystal language model.

Each modality (PXRD / neutron) gets its own small 1D-CNN that maps a
rasterized intensity profile on a fixed 2-theta grid to ``k`` prefix
embeddings. The prefix embeddings replace `<PXRD_EMB>` / `<ND_EMB>`
placeholder tokens inside the condition prompt.

The input is a plain dense vector, so phase-2 experimental realism (noise,
broadening, background) only changes the data augmentation, never this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumCNN(nn.Module):
    def __init__(self, n_embd: int, k_slots: int = 16, channels: int = 128) -> None:
        super().__init__()
        self.k_slots = k_slots
        c = channels
        self.net = nn.Sequential(
            nn.Conv1d(1, c // 4, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(c // 4, c // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(c // 2, c, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(k_slots)
        self.proj = nn.Linear(c, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] rasterized spectrum -> [B, k_slots, n_embd]."""
        h = self.net(x[:, None, :])
        h = self.pool(h)            # [B, C, k]
        h = h.transpose(1, 2)       # [B, k, C]
        return self.proj(h)


class SpectraEncoder(nn.Module):
    """Holds one CNN per supported modality."""

    MODALITIES = ("pxrd", "neutron")

    def __init__(self, n_embd: int, k_slots: int = 16) -> None:
        super().__init__()
        self.k_slots = k_slots
        self.encoders = nn.ModuleDict(
            {name: SpectrumCNN(n_embd, k_slots) for name in self.MODALITIES}
        )

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.encoders[name](x)


def rasterize_spectrum(
    xs: list[float],
    ys: list[float],
    grid_min: float = 5.0,
    grid_max: float = 80.0,
    step: float = 0.05,
    sigma: float = 0.15,
) -> torch.Tensor:
    """Splat discrete peaks onto a fixed dense grid with Gaussian broadening."""
    n = int(round((grid_max - grid_min) / step)) + 1
    grid = torch.arange(n, dtype=torch.float32) * step + grid_min
    vec = torch.zeros(n, dtype=torch.float32)
    if not xs or not ys:
        return vec
    pos = torch.tensor(xs, dtype=torch.float32)
    inten = torch.tensor(ys, dtype=torch.float32)
    keep = inten > 0
    if not bool(keep.any()):
        return vec
    pos, inten = pos[keep], inten[keep]
    inv = 1.0 / (2.0 * sigma * sigma)
    vec = (inten[:, None] * torch.exp(-((grid[None, :] - pos[:, None]) ** 2) * inv)).sum(dim=0)
    peak = float(vec.max())
    if peak > 0:
        vec = vec / peak
    return vec


__all__ = ["SpectraEncoder", "SpectrumCNN", "rasterize_spectrum"]
