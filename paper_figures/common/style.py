"""Shared Nature-style visual language for the manuscript figures.

The palette and layout defaults follow the restrained, schematic-led guidance in
Yuan1z0825/nature-skills.  Every figure remains independently reproducible, but
using this module keeps colors, typography, panel labels, and exports consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "blue_pale": "#DCE8F4",
    "teal": "#42949E",
    "teal_pale": "#D8ECEE",
    "violet": "#9A4D8E",
    "violet_pale": "#EADCE8",
    "green": "#70B77E",
    "green_pale": "#DDF3DE",
    "red": "#B64342",
    "red_pale": "#F6CFCB",
    "gold": "#C79A2B",
    "neutral_0": "#FFFFFF",
    "neutral_1": "#F4F5F6",
    "neutral_2": "#CFCECE",
    "neutral_3": "#8A8A8A",
    "neutral_4": "#4D4D4D",
    "black": "#272727",
}

CONDITION_COLORS = {
    "PXRD": PALETTE["neutral_3"],
    "PXRD + ND": PALETTE["teal"],
    "PXRD + XRF": PALETTE["violet"],
    "PXRD + ND + XRF": PALETTE["blue"],
    "Formula reference": PALETTE["neutral_4"],
}


def apply_style() -> None:
    """Set publication-oriented matplotlib defaults."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    """Place a small bold lowercase panel label."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )


def quiet_axis(ax, *, grid_axis: str | None = None) -> None:
    """Apply restrained axes with an optional light guide grid."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["neutral_3"])
    ax.spines["bottom"].set_color(PALETTE["neutral_3"])
    ax.tick_params(color=PALETTE["neutral_3"], labelcolor=PALETTE["neutral_4"])
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color=PALETTE["neutral_2"],
            linewidth=0.55,
            alpha=0.45,
            zorder=0,
        )


def add_bar_labels(
    ax,
    bars: Iterable,
    *,
    fmt: str = "{:.1f}",
    offset: float = 0.8,
    fontsize: float = 6.5,
    color: str | None = None,
) -> None:
    """Annotate vertical bars without adding a grid-heavy table."""

    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color or PALETTE["neutral_4"],
        )


def add_horizontal_labels(
    ax,
    bars: Iterable,
    *,
    fmt: str = "{:.1f}",
    offset: float = 0.5,
    fontsize: float = 6.5,
) -> None:
    for bar in bars:
        value = float(bar.get_width())
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=PALETTE["neutral_4"],
        )


def draw_peak_pattern(
    ax,
    *,
    color: str,
    peaks: list[tuple[float, float]],
    linewidth: float = 1.2,
    alpha: float = 1.0,
) -> None:
    """Draw a compact schematic diffraction profile from specified peaks."""

    x = np.linspace(5.0, 80.0, 700)
    y = np.zeros_like(x)
    for position, intensity in peaks:
        y += intensity * np.exp(-0.5 * ((x - position) / 0.55) ** 2)
    if y.max() > 0:
        y /= y.max()
    ax.plot(x, y, color=color, lw=linewidth, alpha=alpha)
    ax.set_xlim(5, 80)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def source_note(fig, text: str, *, y: float = 0.008) -> None:
    fig.text(
        0.01,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["neutral_3"],
    )


def save_figure(fig, output_dir: str | Path, stem: str, *, dpi: int = 300) -> None:
    """Save editable vector copies plus the requested high-resolution PNG."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")

