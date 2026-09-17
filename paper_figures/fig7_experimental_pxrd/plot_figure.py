#!/usr/bin/env python3
"""Render manuscript Figure 7: experimental PXRD validation in three layers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.transforms import Bbox


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_DIR = FIGURE_DIR / "outputs"
STEM = "figure_7_experimental_pxrd"

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
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_data.py first.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def panel_header(ax: Any, label: str, title: str, subtitle: str, *, x: float = -0.08) -> None:
    panel_label(ax, label, x=x, y=1.08)
    ax.text(
        0.0,
        1.105,
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
        1.035,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=PALETTE["neutral_3"],
        clip_on=False,
    )


def minus(value: int) -> str:
    return f"{value:+d}".replace("-", "−")


def panel_a_rps(ax: Any, rows: list[dict[str, str]]) -> None:
    panel_header(
        ax,
        "a",
        "Head-to-head with RealPXRD-Solver",
        "PXRD + formula + known lattice • refine-before-match • 95% CI on this work • CNRS tied; RRUFF Top-20 tied after refine",
    )
    order = [
        ("CNRS", "Top-1"),
        ("CNRS", "Top-20"),
        ("RRUFF", "Top-1"),
        ("RRUFF", "Top-20"),
    ]
    indexed = {(row["set"], row["metric"]): row for row in rows}
    x = np.arange(len(order), dtype=float)
    width = 0.24
    series = [
        ("ours", "This work", PALETTE["blue"]),
        ("rps_repro", "RPS reproduced", PALETTE["teal"]),
        ("rps_reported", "RPS reported", PALETTE["neutral_3"]),
    ]
    for offset, (key, _label, color) in zip((-width, 0.0, width), series):
        heights = [int(indexed[item][key]) for item in order]
        bars = ax.bar(
            x + offset,
            heights,
            width=width * 0.92,
            color=color,
            edgecolor="none",
            zorder=3,
        )
        if key == "ours":
            lo = np.array([int(indexed[item]["ours_ci_low"]) for item in order])
            hi = np.array([int(indexed[item]["ours_ci_high"]) for item in order])
            y = np.array(heights, dtype=float)
            ax.errorbar(
                x + offset,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt="none",
                ecolor=PALETTE["black"],
                elinewidth=0.7,
                capsize=2.0,
                capthick=0.7,
                zorder=4,
            )
        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (2.0 if value < 200 else -11),
                str(value),
                ha="center",
                va="bottom" if value < 200 else "top",
                fontsize=6.0,
                color=PALETTE["neutral_4"] if value < 200 else "white",
                fontweight="bold" if value >= 200 else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "CNRS\nTop-1\n$n$=126",
            "CNRS\nTop-20\n$n$=126",
            "RRUFF\nTop-1\n$n$=269",
            "RRUFF\nTop-20\n$n$=269",
        ]
    )
    ax.set_ylabel("Structures matched")
    ax.set_ylim(0, 305)
    ax.set_yticks([0, 100, 200, 269])
    quiet_axis(ax, grid_axis="y")
    ax.axhline(269, color=PALETTE["neutral_2"], lw=0.5, linestyle=(0, (2, 2)), zorder=1)
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in series],
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.0, 1.02),
        columnspacing=0.9,
        handletextpad=0.4,
        fontsize=6.3,
    )
    rruff_t1 = indexed[("RRUFF", "Top-1")]
    ax.annotate(
        f"Δ {minus(int(rruff_t1['diff']))}   $p$=0.003",
        xy=(2.0 - width, int(rruff_t1["ours"])),
        xytext=(2.0, 278),
        ha="center",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color=PALETTE["blue"],
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["blue"],
            "lw": 0.5,
        },
    )


def panel_b_337(ax: Any, rows: list[dict[str, str]], steps: list[dict[str, str]]) -> None:
    panel_header(
        ax,
        "b",
        "Public experimental PXRD, no formula",
        "N=337 measured Cu Kα • ND/XRF are CIF-simulated • counts, not an accuracy rate",
        x=-0.10,
    )
    spec = [row for row in rows if row["protocol"] == "chgnet_union_mace"]
    formula = next(row for row in rows if row["condition_key"] == "pxrd_formula")
    labels = [row["condition_label"].replace("PXRD + ", "+") if row["condition_key"] != "pxrd" else "PXRD" for row in spec] + [
        "Formula"
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    t1 = [int(row["top1"]) for row in spec] + [int(formula["top1"])]
    t20 = [int(row["top20"]) for row in spec] + [int(formula["top20"])]
    bars1 = ax.bar(
        x - width / 2,
        t1,
        width=width * 0.92,
        color=[CONDITION_COLORS.get(row["condition_label"], PALETTE["neutral_4"]) for row in spec]
        + [PALETTE["neutral_2"]],
        edgecolor="none",
        zorder=3,
        label="Top-1",
    )
    bars20 = ax.bar(
        x + width / 2,
        t20,
        width=width * 0.92,
        color=PALETTE["blue_pale"],
        edgecolor=PALETTE["blue"],
        linewidth=0.6,
        zorder=3,
        label="Top-20",
    )
    bars1[-1].set_hatch("///")
    bars1[-1].set_edgecolor(PALETTE["neutral_3"])
    bars1[-1].set_linewidth(0.5)
    bars20[-1].set_hatch("///")
    bars20[-1].set_facecolor(PALETTE["neutral_1"])
    bars20[-1].set_edgecolor(PALETTE["neutral_3"])

    lo = [int(row["top1_ci_low"]) for row in spec]
    hi = [int(row["top1_ci_high"]) for row in spec]
    y = np.array(t1[:4], dtype=float)
    ax.errorbar(
        x[:4] - width / 2,
        y,
        yerr=np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)]),
        fmt="none",
        ecolor=PALETTE["black"],
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        zorder=4,
    )
    t1_label_y = [
        max(value + 1.5, int(hi[i]) + 1.8) if i < len(hi) else value + 1.5
        for i, value in enumerate(t1)
    ]
    for bar, value, y_text in zip(bars1, t1, t1_label_y):
        if value == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            str(value),
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=PALETTE["neutral_4"],
        )
    for bar, value in zip(bars20, t20):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            str(value),
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=PALETTE["neutral_4"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Structures matched (of 337)")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 70, 90])
    quiet_axis(ax, grid_axis="y")
    ax.axvline(3.5, color=PALETTE["neutral_2"], lw=0.6, linestyle=(0, (2, 2)))
    ax.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.0, 0.98),
        fontsize=6.3,
        handletextpad=0.4,
    )
    step_xrf = next(s for s in steps if s["step"].startswith("PXRD"))
    step_nd = next(s for s in steps if s["step"].startswith("+XRF"))
    ax.text(
        0.55,
        42,
        (
            f"PXRD → +XRF\nTop-1 {minus(int(step_xrf['delta']))} "
            f"[{minus(int(step_xrf['ci_low']))}, {minus(int(step_xrf['ci_high']))}]\n"
            f"+XRF → +ND+XRF\n{minus(int(step_nd['delta']))} "
            f"[{minus(int(step_nd['ci_low']))}, {minus(int(step_nd['ci_high']))}]  n.s."
        ),
        ha="center",
        va="bottom",
        fontsize=6.1,
        color=PALETTE["neutral_4"],
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": PALETTE["neutral_2"],
            "linewidth": 0.6,
        },
        zorder=5,
    )
    ax.text(
        4.0,
        -18,
        "unrefined fusion",
        ha="center",
        va="top",
        fontsize=5.6,
        color=PALETTE["neutral_3"],
        clip_on=False,
    )


def panel_c_licoo2(ax: Any, ranks: list[dict[str, str]]) -> None:
    panel_header(
        ax,
        "c",
        "Laboratory LiCoO$_2$",
        "Measured Cu Kα PXRD • ND from the paired CIF • known lattice",
        x=-0.10,
    )
    ax.set_axis_off()
    spec = ax.inset_axes([0.02, 0.08, 0.58, 0.82])
    spectra: dict[str, list[tuple[float, float]]] = {"experimental_pxrd": [], "simulated_nd": []}
    with (DATA_DIR / "licoo2_spectra.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            spectra[raw["series"]].append((float(raw["two_theta_deg"]), float(raw["intensity"])))
    pxrd = np.array(spectra["experimental_pxrd"])
    nd = np.array(spectra["simulated_nd"])
    pxrd_y = pxrd[:, 1] / max(pxrd[:, 1].max(), 1e-9)
    spec.plot(pxrd[:, 0], pxrd_y, color=PALETTE["blue"], lw=0.85, label="Experimental PXRD")
    nd_y = nd[:, 1] / max(nd[:, 1].max(), 1e-9)
    spec.vlines(nd[:, 0], 0.0, nd_y, colors=PALETTE["teal"], linewidth=1.05, alpha=0.85, label="Simulated ND (CIF)")
    spec.set_xlim(10, 80)
    spec.set_ylim(0, 1.18)
    spec.set_xlabel(r"2$\theta$ (deg, Cu K$\alpha$)", fontsize=6.5)
    spec.set_ylabel("Intensity (arb.)", fontsize=6.5)
    spec.set_yticks([])
    quiet_axis(spec)
    spec.legend(
        loc="upper right",
        fontsize=5.8,
        handlelength=1.4,
        borderaxespad=0.1,
    )

    board = ax.inset_axes([0.64, 0.10, 0.34, 0.78])
    board.set_xlim(0, 1)
    board.set_ylim(0, 1)
    board.axis("off")
    indexed = {(row["model"], row["condition"]): row for row in ranks}
    board.text(0.50, 0.96, "Known lattice", ha="center", va="top", fontsize=6.4, fontweight="bold", color=PALETTE["black"])
    board.text(0.27, 0.84, "PXRD", ha="center", fontsize=5.8, color=PALETTE["neutral_4"])
    board.text(0.73, 0.84, "+ ND", ha="center", fontsize=5.8, color=PALETTE["neutral_4"])
    board.text(0.04, 0.62, "Clean", ha="left", va="center", fontsize=5.8, color=PALETTE["neutral_4"])
    board.text(0.04, 0.32, "Noisy", ha="left", va="center", fontsize=5.8, color=PALETTE["neutral_4"])

    def paint(x0: float, y0: float, row: dict[str, str]) -> None:
        hit = row["bucket"] == "T1"
        miss = row["bucket"] == "miss"
        if hit:
            face, edge, text, txt_c = PALETTE["green_pale"], PALETTE["green"], "Top-1", PALETTE["green"]
        elif miss:
            face, edge, text, txt_c = PALETTE["red_pale"], PALETTE["red"], "miss", PALETTE["red"]
        else:
            face, edge, text, txt_c = PALETTE["gold"], PALETTE["gold"], f"T{row['best_k']}", PALETTE["black"]
            face = PALETTE["neutral_1"]
        board.add_patch(
            FancyBboxPatch(
                (x0, y0),
                0.40,
                0.20,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.7,
            )
        )
        board.text(x0 + 0.20, y0 + 0.10, text, ha="center", va="center", fontsize=7.0, fontweight="bold", color=txt_c)

    paint(0.08, 0.52, indexed[("clean", "pxrd")])
    paint(0.54, 0.52, indexed[("clean", "pxrd_neutron")])
    paint(0.08, 0.22, indexed[("noisy", "pxrd")])
    paint(0.54, 0.22, indexed[("noisy", "pxrd_neutron")])
    board.annotate(
        "",
        xy=(0.74, 0.32),
        xytext=(0.28, 0.32),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["teal"], "lw": 0.9},
    )
    board.text(
        0.50,
        0.06,
        "ND lifts rank 2 → 1",
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=PALETTE["teal"],
    )


def main() -> None:
    apply_style()
    rps = load_csv("fig7a_rps_head_to_head.csv")
    public = load_csv("fig7b_public337_known_lattice.csv")
    steps = load_csv("fig7b_paired_steps.csv")
    ranks = load_csv("fig7c_licoo2_known_ranks.csv")

    fig = plt.figure(figsize=(7.2, 8.55))
    grid = fig.add_gridspec(
        3,
        1,
        left=0.10,
        right=0.985,
        bottom=0.068,
        top=0.88,
        hspace=0.70,
    )
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    ax_c = fig.add_subplot(grid[2])

    panel_a_rps(ax_a, rps)
    panel_b_337(ax_b, public, steps)
    panel_c_licoo2(ax_c, ranks)

    fig.text(
        0.10,
        0.975,
        "Fig. 7 | Experimental powder diffraction",
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    fig.text(
        0.10,
        0.948,
        (
            "a, Same protocol as RealPXRD-Solver (formula + known lattice). "
            "b, Multimodal ladder on measured Cu Kα without formula. "
            "c, One laboratory case."
        ),
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["neutral_4"],
    )
    source_note(
        fig,
        (
            "Sources: rps_*_bootstrap_refined.json; ladder_public_si_report_spec_cm.json "
            "(spectroscopic rungs: CHGNet ∪ MACE refine). Formula reference in b is unrefined fusion. "
            "LiCoO2 ND is CIF-simulated."
        ),
        y=0.012,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox = Bbox.from_bounds(0, 0, *fig.get_size_inches())
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUTPUT_DIR / f"{STEM}.{suffix}", bbox_inches=bbox, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUTPUT_DIR / f"{STEM}.png")


if __name__ == "__main__":
    main()
