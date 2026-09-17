#!/usr/bin/env python3
"""Render manuscript Figure 6: robustness to synthetic experimental artefacts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_DIR = FIGURE_DIR / "outputs"
STEM = "figure_6_noise_robustness"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_figures.common.style import (  # noqa: E402
    PALETTE,
    apply_style,
    panel_label,
    quiet_axis,
    source_note,
)


CONDITION_ORDER = ("pxrd_xrf", "pxrd_neutron_xrf", "pxrd_formula")
SHORT_LABELS = {
    "pxrd_xrf": "PXRD + XRF",
    "pxrd_neutron_xrf": "PXRD + ND + XRF",
    "pxrd_neutron_xrf_xps": "PXRD + ND + XRF + XPS",
    "pxrd_formula": "Formula ref.",
}


def load_csv(filename: str) -> list[dict[str, str]]:
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Curated data not found: {path}. Run prepare_data.py first."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def panel_header(
    ax: Any, label: str, title: str, subtitle: str, *, label_x: float = -0.11
) -> None:
    panel_label(ax, label, x=label_x, y=1.085)
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
        fontsize=6.5,
        color=PALETTE["neutral_3"],
        clip_on=False,
    )


def gaussian_profile(
    x: np.ndarray,
    positions: np.ndarray,
    widths: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    profile = np.zeros_like(x)
    for position, width, amplitude in zip(
        positions, widths, amplitudes, strict=True
    ):
        profile += amplitude * np.exp(
            -0.5 * np.square((x - position) / width)
        )
    return profile


def draw_trace_card(
    ax: Any,
    x0: float,
    title: str,
    profile: np.ndarray,
    *,
    color: str,
    width: float = 0.138,
) -> None:
    y0, height = 0.27, 0.47
    card = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=PALETTE["neutral_1"],
        edgecolor=PALETTE["neutral_2"],
        linewidth=0.65,
    )
    ax.add_patch(card)
    x = np.linspace(0.0, 1.0, profile.size)
    clipped = np.clip(profile, 0.0, None)
    maximum = float(clipped.max())
    if maximum:
        clipped = clipped / maximum
    trace_x = x0 + 0.010 + x * (width - 0.020)
    trace_y = y0 + 0.055 + clipped * (height - 0.135)
    ax.plot(trace_x, trace_y, color=color, lw=1.05, clip_on=True)
    ax.plot(
        [x0 + 0.010, x0 + width - 0.010],
        [y0 + 0.050, y0 + 0.050],
        color=PALETTE["neutral_2"],
        lw=0.45,
    )
    ax.text(
        x0 + width / 2,
        y0 + height + 0.035,
        title,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["neutral_4"],
        linespacing=0.95,
    )


def panel_a_schematic(ax: Any) -> None:
    panel_header(
        ax,
        "a",
        "Synthetic experimental artefacts",
        "Online perturbation workflow • schematic, qualitative, not to scale",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    x = np.linspace(0.0, 1.0, 240)
    positions = np.array([0.16, 0.37, 0.57, 0.77, 0.89])
    amplitudes = np.array([0.42, 1.00, 0.63, 0.78, 0.38])
    clean = gaussian_profile(
        x, positions, np.full(positions.shape, 0.012), amplitudes
    )
    shifted = gaussian_profile(
        x, positions + 0.034, np.full(positions.shape, 0.012), amplitudes
    )
    broadened = gaussian_profile(
        x, positions + 0.034, np.full(positions.shape, 0.032), amplitudes
    )
    textured = gaussian_profile(
        x,
        positions + 0.034,
        np.full(positions.shape, 0.032),
        amplitudes * np.array([1.35, 0.62, 1.18, 0.52, 1.42]),
    )
    background = textured + 0.10 + 0.13 * x + 0.06 * np.square(x - 0.42)
    rng = np.random.default_rng(6)
    noisy = np.clip(
        background + rng.normal(0.0, 0.052, size=x.size), 0.0, None
    )

    cards = [
        ("Clean\nsimulation", clean, PALETTE["blue"]),
        ("2θ shift", shifted, PALETTE["violet"]),
        ("Peak\nbroadening", broadened, PALETTE["violet"]),
        ("Intensity /\ntexture", textured, PALETTE["teal"]),
        ("Background", background, PALETTE["gold"]),
        ("Counting\nnoise", noisy, PALETTE["red"]),
    ]
    card_x = np.linspace(0.010, 0.852, len(cards))
    for index, (x0, title, profile, color) in enumerate(
        (x0, *card) for x0, card in zip(card_x, cards, strict=True)
    ):
        draw_trace_card(ax, x0, title, profile, color=color)
        if index < len(cards) - 1:
            arrow = FancyArrowPatch(
                (x0 + 0.140, 0.505),
                (card_x[index + 1] - 0.006, 0.505),
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.65,
                color=PALETTE["neutral_3"],
            )
            ax.add_patch(arrow)

    note = FancyBboxPatch(
        (0.185, 0.055),
        0.630,
        0.095,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=PALETTE["red_pale"],
        edgecolor="none",
        alpha=0.75,
    )
    ax.add_patch(note)
    ax.text(
        0.500,
        0.102,
        "NOISY = synthetic perturbations of simulated inputs",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=PALETTE["red"],
    )


def clean_noisy_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["condition"], row["lattice"]): row for row in rows}


def panel_clean_noisy(
    ax: Any,
    rows: list[dict[str, str]],
    *,
    label: str,
    metric: str,
    title: str,
    subtitle: str,
    xlim: tuple[float, float],
    xticks: list[float],
    highlight_recall: bool = False,
) -> None:
    panel_header(ax, label, title, subtitle)
    indexed = clean_noisy_index(rows)
    y_centers = np.array([2.0, 1.0, 0.0])
    offsets = {"free": 0.15, "known": -0.15}
    markers = {"free": "o", "known": "s"}

    if highlight_recall:
        ax.axvspan(
            92.4,
            100.2,
            color=PALETTE["green_pale"],
            alpha=0.75,
            zorder=0,
        )

    for y_center, condition in zip(
        y_centers, CONDITION_ORDER, strict=True
    ):
        for lattice in ("free", "known"):
            row = indexed[(condition, lattice)]
            y = y_center + offsets[lattice]
            clean = number(row, f"clean_{metric}")
            noisy = number(row, f"noisy_{metric}")
            ax.plot(
                [clean, noisy],
                [y, y],
                color=PALETTE["neutral_2"],
                lw=1.15,
                zorder=2,
            )
            ax.scatter(
                clean,
                y,
                marker=markers[lattice],
                s=25,
                facecolor="white",
                edgecolor=PALETTE["neutral_4"],
                linewidth=0.9,
                zorder=3,
            )
            ax.scatter(
                noisy,
                y,
                marker=markers[lattice],
                s=25,
                facecolor=PALETTE["red"],
                edgecolor=PALETTE["red"],
                linewidth=0.9,
                zorder=4,
            )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y_centers)
    ax.set_yticklabels([SHORT_LABELS[item] for item in CONDITION_ORDER])
    ax.set_xlabel(f"{metric.replace('T', 'Top-')} match rate (%)")
    quiet_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)

    if metric == "T1":
        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=PALETTE["neutral_4"],
                markersize=4.5,
                label="Clean",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PALETTE["red"],
                markeredgecolor=PALETTE["red"],
                markersize=4.5,
                label="Noisy",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PALETTE["neutral_2"],
                markeredgecolor=PALETTE["neutral_2"],
                markersize=4.5,
                label="Lattice-free",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor=PALETTE["neutral_2"],
                markeredgecolor=PALETTE["neutral_2"],
                markersize=4.2,
                label="Known lattice",
            ),
        ]
        ax.legend(
            handles=legend,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.35,
            borderaxespad=0.0,
            fontsize=6.0,
        )
    else:
        noisy_values = [
            number(indexed[(condition, lattice)], f"noisy_{metric}")
            for condition in CONDITION_ORDER
            for lattice in ("free", "known")
        ]
        ax.text(
            0.03,
            0.08,
            f"Noisy Top-20 ≥ {min(noisy_values):.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=PALETTE["green"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": PALETTE["green"],
                "linewidth": 0.7,
            },
        )


def panel_d_deploy(ax: Any, rows: list[dict[str, str]]) -> None:
    panel_header(
        ax,
        "d",
        "Noisy-aware reranking",
        "Same frozen candidates • noisy + lattice-free only",
    )
    indexed = {row["condition"]: row for row in rows}
    y = np.array([2.0, 1.0, 0.0])
    for y_value, condition in zip(y, CONDITION_ORDER, strict=True):
        row = indexed[condition]
        baseline = number(row, "baseline_T1")
        deployed = number(row, "deploy_T1")
        delta = number(row, "delta_T1")
        ax.plot(
            [baseline, deployed],
            [y_value, y_value],
            color=PALETTE["violet_pale"],
            lw=3.5,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            baseline,
            y_value,
            s=32,
            facecolor="white",
            edgecolor=PALETTE["neutral_4"],
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            deployed,
            y_value,
            s=35,
            facecolor=PALETTE["violet"],
            edgecolor=PALETTE["violet"],
            linewidth=0.9,
            zorder=4,
        )
        ax.text(
            deployed + 0.32,
            y_value,
            f"+{delta:.1f}",
            ha="left",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=PALETTE["violet"],
        )

    ax.set_xlim(70.0, 81.2)
    ax.set_xticks([70, 75, 80])
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS[item] for item in CONDITION_ORDER])
    ax.set_xlabel("Top-1 match rate (%)")
    quiet_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.text(
        0.02,
        0.06,
        "σ=0.25  →  σ=0.40 + shift ±0.20°",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=PALETTE["neutral_4"],
    )


def panel_e_refinement(ax: Any, rows: list[dict[str, str]]) -> None:
    panel_header(
        ax,
        "e",
        "Rietveld-lite reranking",
        "Within each refinement pool • noisy + lattice-free • top-25",
    )
    base_order = (
        "pxrd_xrf",
        "pxrd_neutron_xrf",
        "pxrd_neutron_xrf_xps",
        "pxrd_formula",
    )
    base_rows = {
        row["condition"]: row
        for row in rows
        if row["population"] == "base"
    }
    bigcell = next(row for row in rows if row["population"] == "bigcell")
    y_positions = np.array([4.4, 3.4, 2.4, 1.4])

    for y, condition in zip(y_positions, base_order, strict=True):
        row = base_rows[condition]
        baseline = number(row, "baseline_T1")
        refined = number(row, "refined_T1")
        delta = number(row, "delta_T1")
        alpha = 0.38 if condition == "pxrd_neutron_xrf_xps" else 1.0
        ax.plot(
            [baseline, refined],
            [y, y],
            color=PALETTE["blue_pale"],
            lw=3.5,
            solid_capstyle="round",
            alpha=alpha,
            zorder=1,
        )
        ax.scatter(
            baseline,
            y,
            s=30,
            facecolor="white",
            edgecolor=PALETTE["neutral_4"],
            linewidth=0.9,
            alpha=alpha,
            zorder=3,
        )
        ax.scatter(
            refined,
            y,
            s=34,
            facecolor=PALETTE["blue"],
            edgecolor=PALETTE["blue"],
            linewidth=0.9,
            alpha=alpha,
            zorder=4,
        )
        ax.text(
            refined + 0.55,
            y,
            f"+{delta:.1f}",
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=PALETTE["blue"],
            alpha=alpha,
        )

    big_y = 0.0
    big_base = number(bigcell, "baseline_T1")
    big_refined = number(bigcell, "refined_T1")
    big_delta = number(bigcell, "delta_T1")
    ax.axhline(
        0.72,
        color=PALETTE["neutral_2"],
        lw=0.65,
        linestyle=(0, (2, 2)),
    )
    ax.plot(
        [big_base, big_refined],
        [big_y, big_y],
        color=PALETTE["green_pale"],
        lw=3.5,
        solid_capstyle="round",
        zorder=1,
    )
    ax.scatter(
        big_base,
        big_y,
        marker="D",
        s=33,
        facecolor="white",
        edgecolor=PALETTE["green"],
        linewidth=0.9,
        zorder=3,
    )
    ax.scatter(
        big_refined,
        big_y,
        marker="D",
        s=36,
        facecolor=PALETTE["green"],
        edgecolor=PALETTE["green"],
        linewidth=0.9,
        zorder=4,
    )
    ax.text(
        big_refined + 0.55,
        big_y,
        f"+{big_delta:.1f}",
        ha="left",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=PALETTE["green"],
    )

    ax.set_xlim(56.5, 86.5)
    ax.set_xticks([60, 70, 80])
    ax.set_ylim(-0.55, 4.95)
    ax.set_yticks([*y_positions, big_y])
    ax.set_yticklabels(
        ["PXRD + XRF", "PXRD + ND + XRF", "+ XPS", "Formula ref."]
        + ["Big-cell multi"]
    )
    ticklabels = ax.get_yticklabels()
    ticklabels[2].set_color(PALETTE["neutral_3"])
    ax.set_xlabel("Top-1 match rate (%)")
    quiet_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.text(
        0.99,
        0.28,
        "base mean +3.3 pp",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=PALETTE["blue"],
    )


def summary_box(
    ax: Any,
    xy: tuple[float, float],
    *,
    facecolor: str,
    accent: str,
    title: str,
    detail: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        0.375,
        0.175,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor="none",
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.020,
        y + 0.118,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        x + 0.020,
        y + 0.054,
        detail,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.4,
        color=PALETTE["neutral_4"],
        linespacing=1.1,
    )


def panel_f_summary(
    ax: Any,
    bigcell_rows: list[dict[str, str]],
    clean_noisy_rows: list[dict[str, str]],
    deploy_rows: list[dict[str, str]],
    refine_rows: list[dict[str, str]],
) -> None:
    panel_header(
        ax,
        "f",
        "Robust recall, residual ranking gap",
        "Big-cell specialist; within-pool deltas only",
    )
    ax.axis("off")
    indexed = {row["lattice"]: row for row in bigcell_rows}
    chart = ax.inset_axes([0.005, 0.08, 0.575, 0.84])

    chart_rows = [
        (
            "Free · T1",
            number(indexed["free"], "clean_bigcell_T1"),
            number(indexed["free"], "noisy_bigcell_T1"),
            number(indexed["free"], "delta_T1"),
        ),
        (
            "Free · T20",
            number(indexed["free"], "clean_bigcell_T20"),
            number(indexed["free"], "noisy_bigcell_T20"),
            number(indexed["free"], "delta_T20"),
        ),
        (
            "Known · T1",
            number(indexed["known"], "clean_bigcell_T1"),
            number(indexed["known"], "noisy_bigcell_T1"),
            number(indexed["known"], "delta_T1"),
        ),
        (
            "Known · T20",
            number(indexed["known"], "clean_bigcell_T20"),
            number(indexed["known"], "noisy_bigcell_T20"),
            number(indexed["known"], "delta_T20"),
        ),
    ]
    y_positions = np.array([3.0, 2.0, 1.0, 0.0])
    for y, (label, clean, noisy, delta) in zip(
        y_positions, chart_rows, strict=True
    ):
        chart.plot(
            [clean, noisy],
            [y, y],
            color=PALETTE["neutral_2"],
            lw=1.2,
            zorder=1,
        )
        chart.scatter(
            clean,
            y,
            s=29,
            facecolor="white",
            edgecolor=PALETTE["neutral_4"],
            linewidth=0.9,
            zorder=3,
        )
        chart.scatter(
            noisy,
            y,
            s=31,
            facecolor=PALETTE["red"],
            edgecolor=PALETTE["red"],
            linewidth=0.9,
            zorder=4,
        )
        chart.text(
            noisy - 0.65,
            y + 0.23,
            f"{delta:.1f} pp".replace("-", "−"),
            ha="right",
            va="center",
            fontsize=5.9,
            color=PALETTE["red"],
        )
    chart.axhline(
        1.5,
        color=PALETTE["neutral_2"],
        lw=0.6,
        linestyle=(0, (2, 2)),
    )
    chart.set_xlim(53.0, 81.0)
    chart.set_xticks([55, 65, 75])
    chart.set_ylim(-0.55, 3.55)
    chart.set_yticks(y_positions)
    chart.set_yticklabels([item[0] for item in chart_rows])
    chart.set_xlabel("Match rate (%)")
    quiet_axis(chart, grid_axis="x")
    chart.spines["left"].set_visible(False)
    chart.tick_params(axis="y", length=0, pad=3)
    chart.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=PALETTE["neutral_4"],
                markersize=4.3,
                label="Clean",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=PALETTE["red"],
                markeredgecolor=PALETTE["red"],
                markersize=4.3,
                label="Noisy",
            ),
        ],
        loc="lower left",
        ncol=2,
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=0.8,
        fontsize=6.0,
    )

    noisy_t20 = [
        number(row, "noisy_T20") for row in clean_noisy_rows
    ]
    deploy_gain = np.mean([number(row, "delta_T1") for row in deploy_rows])
    base_refine = [
        row for row in refine_rows if row["population"] == "base"
    ]
    refine_gain = np.mean([number(row, "delta_T1") for row in base_refine])

    summary_box(
        ax,
        (0.605, 0.715),
        facecolor=PALETTE["green_pale"],
        accent=PALETTE["green"],
        title="High-info recall",
        detail=f"Noisy Top-20 ≥ {min(noisy_t20):.1f}%\nfree + known",
    )
    summary_box(
        ax,
        (0.605, 0.445),
        facecolor=PALETTE["violet_pale"],
        accent=PALETTE["violet"],
        title="Noise-aware rank",
        detail=f"Free mean +{deploy_gain:.1f} pp\nTop-1",
    )
    summary_box(
        ax,
        (0.605, 0.175),
        facecolor=PALETTE["blue_pale"],
        accent=PALETTE["blue"],
        title="Rietveld-lite",
        detail=f"Base mean +{refine_gain:.1f} pp\nTop-1; recall fixed",
    )


def main() -> None:
    apply_style()
    clean_noisy_rows = load_csv("clean_noisy_high_information.csv")
    deploy_rows = load_csv("deploy_rerank_free.csv")
    refine_rows = load_csv("refinement_free.csv")
    bigcell_rows = load_csv("bigcell_clean_noisy.csv")

    # A tall two-column layout keeps every panel readable at the final
    # Nature-family double-column width (7.2 in) without slide-sized scaling.
    fig = plt.figure(figsize=(7.2, 8.6))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.135,
        right=0.985,
        bottom=0.075,
        top=0.855,
        wspace=0.42,
        hspace=0.58,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(3) for col in range(2)]

    panel_a_schematic(axes[0])
    panel_clean_noisy(
        axes[1],
        clean_noisy_rows,
        label="b",
        metric="T1",
        title="Top-1 under high-information inputs",
        subtitle="Clean ○ / noisy ●; circle free, square known; N=500",
        xlim=(68.0, 100.5),
        xticks=[70, 80, 90, 100],
    )
    panel_clean_noisy(
        axes[2],
        clean_noisy_rows,
        label="c",
        metric="T20",
        title="Top-20 recall is more robust",
        subtitle="Clean ○ / noisy ●; circle free, square known; N=500",
        xlim=(90.0, 100.2),
        xticks=[90, 95, 100],
        highlight_recall=True,
    )
    panel_d_deploy(axes[3], deploy_rows)
    panel_e_refinement(axes[4], refine_rows)
    panel_f_summary(
        axes[5],
        bigcell_rows,
        clean_noisy_rows,
        deploy_rows,
        refine_rows,
    )

    fig.text(
        0.135,
        0.962,
        "Fig. 6 | Robustness to synthetic experimental artefacts",
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    fig.text(
        0.135,
        0.928,
        (
            "“Noisy” = synthetic perturbations of simulated inputs; not real "
            "experimental measurements.\nAll results: N=500/config."
        ),
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["red"],
        fontweight="bold",
        linespacing=1.2,
    )
    fig.text(
        0.135,
        0.904,
        (
            "Clean/noisy ladders compare clean-trained + clean input with "
            "noise-finetuned + noisy input; percentages are match rates."
        ),
        ha="left",
        va="top",
        fontsize=5.8,
        color=PALETTE["neutral_3"],
    )
    source_note(
        fig,
        (
            "Sources: noisy_robustness_summary.json; "
            "noisy_rerank_deploy_base_ladder.json; "
            "refine_rerank_ablation_base_free.json;\n"
            "refine_rerank_ablation_bigcell_free.json. "
            "Deploy and refinement deltas use only their own frozen candidate pools."
        ),
        y=0.018,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_figure_bbox = Bbox.from_bounds(0, 0, *fig.get_size_inches())
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            OUTPUT_DIR / f"{STEM}.{suffix}",
            bbox_inches=full_figure_bbox,
            pad_inches=0,
            **kwargs,
        )
    plt.close(fig)
    print(OUTPUT_DIR / f"{STEM}.png")
    print(OUTPUT_DIR / f"{STEM}.pdf")
    print(OUTPUT_DIR / f"{STEM}.svg")


if __name__ == "__main__":
    main()
