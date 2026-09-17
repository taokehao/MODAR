#!/usr/bin/env python3
"""SI figures S17–S20: public-337 spectroscopic ladder beyond Fig. 7b."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox


SI_DIR = Path(__file__).resolve().parent
REPO_ROOT = SI_DIR.parents[1]
DATA_DIR = SI_DIR / "data"
OUTPUT_DIR = SI_DIR / "outputs"

sys.path.insert(0, str(REPO_ROOT))
from paper_figures.common.style import (  # noqa: E402
    CONDITION_COLORS,
    PALETTE,
    apply_style,
    panel_label,
    quiet_axis,
    source_note,
)


def load_csv(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def header(ax: Any, label: str, title: str, subtitle: str) -> None:
    panel_label(ax, label, x=-0.10, y=1.06)
    ax.text(
        0.0,
        1.08,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=PALETTE["neutral_3"],
        clip_on=False,
    )


def minus(value: int) -> str:
    return f"{value:+d}".replace("-", "−")


def save(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox = Bbox.from_bounds(0, 0, *fig.get_size_inches())
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches=bbox, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUTPUT_DIR / f"{stem}.png")


def plot_s17(rows: list[dict[str, str]]) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.62])
    header(
        ax,
        "a",
        "Public 337, lattice-free spectroscopic ladder",
        "CHGNet ∪ MACE refine • N=337 • ND/XRF CIF-simulated • XPS is SI-only, not in Fig. 7b",
    )
    subset = [row for row in rows if row["lattice"] == "Lattice-free"]
    labels = ["PXRD", "+ND", "+XRF", "+ND+XRF", "+XPS"]
    t1 = [int(row["top1"]) for row in subset]
    t20 = [int(row["top20"]) for row in subset]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    colors = [
        CONDITION_COLORS["PXRD"],
        CONDITION_COLORS["PXRD + ND"],
        CONDITION_COLORS["PXRD + XRF"],
        CONDITION_COLORS["PXRD + ND + XRF"],
        PALETTE["neutral_2"],
    ]
    ax.bar(x - width / 2, t1, width=width * 0.92, color=colors, edgecolor="none", zorder=3, label="Top-1")
    ax.bar(
        x + width / 2,
        t20,
        width=width * 0.92,
        color=PALETTE["blue_pale"],
        edgecolor=PALETTE["blue"],
        linewidth=0.55,
        zorder=3,
        label="Top-20",
    )
    lo = np.array([int(row["top1_ci_low"]) for row in subset], dtype=float)
    hi = np.array([int(row["top1_ci_high"]) for row in subset], dtype=float)
    y = np.array(t1, dtype=float)
    ax.errorbar(
        x - width / 2,
        y,
        yerr=np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)]),
        fmt="none",
        ecolor=PALETTE["black"],
        elinewidth=0.65,
        capsize=1.8,
        zorder=4,
    )
    for i, (a, b) in enumerate(zip(t1, t20)):
        ax.text(
            i - width / 2,
            max(a, int(hi[i])) + 1.4,
            str(a),
            ha="center",
            fontsize=6.0,
            color=PALETTE["neutral_4"],
        )
        ax.text(i + width / 2, b + 1.2, str(b), ha="center", fontsize=6.0, color=PALETTE["neutral_4"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Structures matched (of 337)")
    ax.set_ylim(0, 78)
    quiet_axis(ax, grid_axis="y")
    ax.legend(loc="upper left", ncol=2, fontsize=6.4)
    ax.text(
        0.99,
        0.90,
        "+XRF → +ND+XRF  Top-1 +15 [+6, +24]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=PALETTE["teal"],
    )
    fig.text(
        0.10,
        0.96,
        "Fig. S17 | Lattice-free public-337 ladder",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=PALETTE["black"],
    )
    source_note(
        fig,
        "Source: ladder_public_si_report_spec_cm.json. Known-lattice counterpart is Fig. 7b. XPS is not a main-text condition.",
        y=0.03,
    )
    save(fig, "fig_s17_public337_lattice_free")


def plot_s18(rows: list[dict[str, str]]) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.55))
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.62])
    header(
        ax,
        "a",
        "Known-lattice public 337, all cells versus n_sites ≤ 20",
        "Same refined spectroscopic rungs as Fig. 7b • n≤20 is 122 / 337",
    )
    labels = ["PXRD", "+ND", "+XRF", "+ND+XRF"]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    t1_all = [int(row["top1_all"]) for row in rows]
    t1_small = [int(row["top1_le20"]) for row in rows]
    ax.bar(x - width / 2, t1_all, width=width * 0.92, color=PALETTE["blue"], edgecolor="none", label="All cells (337) Top-1")
    ax.bar(
        x + width / 2,
        t1_small,
        width=width * 0.92,
        facecolor="white",
        edgecolor=PALETTE["blue"],
        linewidth=0.9,
        label="n_sites ≤ 20 (122) Top-1",
    )
    for i, (a, b) in enumerate(zip(t1_all, t1_small)):
        if a > 0:
            ax.text(i - width / 2, a + 1.4, f"{a}/337", ha="center", fontsize=5.8, color=PALETTE["neutral_4"])
        if b > 0:
            ax.text(i + width / 2, b + 1.4, f"{b}/122", ha="center", fontsize=5.8, color=PALETTE["neutral_4"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-1 structures matched")
    ax.set_ylim(0, 98)
    quiet_axis(ax, grid_axis="y")
    ax.legend(loc="upper left", fontsize=6.3)
    fig.text(
        0.10,
        0.96,
        "Fig. S18 | Public 337 size split",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=PALETTE["black"],
    )
    source_note(
        fig,
        "Source: ladder_public_si_report_spec_cm.json. n≤20 is closer to the CNRS/RRUFF size band but still has no formula.",
        y=0.03,
    )
    save(fig, "fig_s18_public337_size_split")


def plot_s19(rows: list[dict[str, str]]) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_axes([0.28, 0.14, 0.68, 0.68])
    header(
        ax,
        "a",
        "Paired ladder steps on public 337",
        "Paired bootstrap B=20,000 • Δ count of matched structures • 95% CI",
    )
    wanted = [row for row in rows if row["metric"] == "Top-1"]
    order = [
        ("Known lattice", "PXRD → +XRF"),
        ("Lattice-free", "PXRD → +XRF"),
        ("Known lattice", "+XRF → +ND+XRF"),
        ("Lattice-free", "+XRF → +ND+XRF"),
        ("Known lattice", "PXRD → +ND"),
        ("Lattice-free", "PXRD → +ND"),
    ]
    indexed = {(row["lattice"], row["step"]): row for row in wanted}
    y = np.arange(len(order), dtype=float)[::-1]
    colors = {
        "PXRD → +XRF": PALETTE["violet"],
        "+XRF → +ND+XRF": PALETTE["teal"],
        "PXRD → +ND": PALETTE["neutral_3"],
    }
    for yi, (lattice, step) in zip(y, order):
        row = indexed[(lattice, step)]
        delta, lo, hi = int(row["delta"]), int(row["ci_low"]), int(row["ci_high"])
        color = colors[step]
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.4, zorder=2)
        facecolor = color if lattice == "Known lattice" else "white"
        ax.scatter(
            [delta],
            [yi],
            s=28,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
            marker="s" if lattice == "Known lattice" else "o",
        )
        ax.text(
            hi + 2.2,
            yi,
            f"{minus(delta)} [{minus(lo)}, {minus(hi)}]",
            ha="left",
            va="center",
            fontsize=6.0,
            color=PALETTE["neutral_4"],
        )
    ax.axvline(0, color=PALETTE["neutral_2"], lw=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{step}\n{lattice}" for lattice, step in order])
    ax.set_xlabel("Δ matched structures (Top-1)")
    ax.set_xlim(-8, 100)
    quiet_axis(ax, grid_axis="x")
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=PALETTE["neutral_4"], label="Lattice-free"),
            Patch(facecolor=PALETTE["neutral_4"], edgecolor=PALETTE["neutral_4"], label="Known lattice"),
        ],
        loc="lower right",
        fontsize=6.2,
    )
    fig.text(
        0.10,
        0.96,
        "Fig. S19 | Public-337 paired increments",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=PALETTE["black"],
    )
    source_note(
        fig,
        "Source: ladder_public_si_report_spec_cm.json. The known-lattice +XRF → +ND+XRF interval includes 0.",
        y=0.025,
    )
    save(fig, "fig_s19_public337_paired_steps")


def plot_s20(rows: list[dict[str, str]]) -> None:
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.5))
    ax = fig.add_axes([0.10, 0.18, 0.86, 0.62])
    header(
        ax,
        "a",
        "XPS after ND+XRF on public 337",
        "Same CHGNet ∪ MACE refine as Fig. 7b • XPS is withheld from the main-text ladder",
    )
    groups = ["Known lattice", "Lattice-free"]
    x = np.arange(len(groups), dtype=float)
    width = 0.32
    indexed = {(row["lattice"], row["condition_key"]): row for row in rows}
    base = [indexed[(g, "pxrd_neutron_xrf")] for g in groups]
    xps = [indexed[(g, "pxrd_neutron_xrf_xps")] for g in groups]
    t1_base = [int(row["top1"]) for row in base]
    t1_xps = [int(row["top1"]) for row in xps]
    ax.bar(
        x - width / 2,
        t1_base,
        width=width * 0.92,
        color=PALETTE["blue"],
        edgecolor="none",
        zorder=3,
        label="+ND+XRF",
    )
    ax.bar(
        x + width / 2,
        t1_xps,
        width=width * 0.92,
        facecolor="white",
        edgecolor=PALETTE["neutral_3"],
        linewidth=0.9,
        zorder=3,
        label="+ND+XRF+XPS",
    )
    for i, row in enumerate(base):
        lo, hi, y = int(row["top1_ci_low"]), int(row["top1_ci_high"]), t1_base[i]
        ax.errorbar(
            x[i] - width / 2,
            y,
            yerr=[[max(0, y - lo)], [max(0, hi - y)]],
            fmt="none",
            ecolor=PALETTE["black"],
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
        )
    for i, row in enumerate(xps):
        lo, hi, y = int(row["top1_ci_low"]), int(row["top1_ci_high"]), t1_xps[i]
        ax.errorbar(
            x[i] + width / 2,
            y,
            yerr=[[max(0, y - lo)], [max(0, hi - y)]],
            fmt="none",
            ecolor=PALETTE["black"],
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
        )
    for i, (a, b, base_row, xps_row) in enumerate(zip(t1_base, t1_xps, base, xps)):
        ax.text(
            x[i] - width / 2,
            max(a, int(base_row["top1_ci_high"])) + 2.2,
            str(a),
            ha="center",
            fontsize=6.2,
            color=PALETTE["neutral_4"],
        )
        ax.text(
            x[i] + width / 2,
            max(b, int(xps_row["top1_ci_high"])) + 2.2,
            str(b),
            ha="center",
            fontsize=6.2,
            color=PALETTE["neutral_4"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Top-1 structures matched (of 337)")
    ax.set_ylim(0, 120)
    quiet_axis(ax, grid_axis="y")
    ax.legend(loc="upper right", fontsize=6.4)
    fig.text(
        0.10,
        0.96,
        "Fig. S20 | XPS increment on public 337",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color=PALETTE["black"],
    )
    source_note(
        fig,
        "Source: ladder_public_si_report_spec_cm.json. Adding XPS does not move Top-1 outside the +ND+XRF interval.",
        y=0.03,
    )
    save(fig, "fig_s20_public337_xps")


def main() -> None:
    plot_s17(load_csv("fig_s17_public337_ladder.csv"))
    plot_s18(load_csv("fig_s18_public337_size_split.csv"))
    plot_s19(load_csv("fig_s19_public337_paired_steps.csv"))
    plot_s20(load_csv("fig_s20_public337_xps.csv"))


if __name__ == "__main__":
    main()
