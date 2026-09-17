#!/usr/bin/env python3
"""Render the Nature-style Figure 2 dataset atlas from prepared aggregates."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
sys.path.insert(0, str(REPO_ROOT))

from paper_figures.common.style import (  # noqa: E402
    PALETTE,
    apply_style,
    panel_label,
    quiet_axis,
    save_figure,
    source_note,
)


def read_csv(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"prepared data missing: {path}; run prepare_data.py first"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def title_with_subtitle(ax, title: str, subtitle: str, *, title_y: float = 1.08) -> None:
    ax.text(
        0,
        title_y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )
    ax.text(
        0,
        title_y - 0.075,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["neutral_3"],
        clip_on=False,
    )


def draw_panel_a(ax, funnel: list[dict[str, str]], splits: list[dict[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1)
    title_with_subtitle(
        ax,
        "Dataset governance funnel",
        "Unified corpus → Tier-A filter → tight deduplication → frozen split",
        title_y=1.04,
    )
    panel_label(ax, "a", x=-0.075, y=1.035)

    labels = ["Unified\nrecords", "Tier-A\nquality", "Tight\ndedup", "Frozen\nsplit"]
    colors = [
        PALETTE["neutral_4"],
        PALETTE["teal"],
        PALETTE["blue_2"],
        PALETTE["blue"],
    ]
    centers = [0.105, 0.365, 0.625, 0.885]
    widths = [0.205, 0.195, 0.195, 0.205]
    raw_count = int(funnel[0]["count"])
    counts = [int(row["count"]) for row in funnel]
    heights = [0.62, 0.42, 0.38, 0.38]
    center_y = 0.51

    for index, (center, width, height, count, color, label) in enumerate(
        zip(centers, widths, heights, counts, colors, labels)
    ):
        box = FancyBboxPatch(
            (center - width / 2, center_y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            center,
            center_y + 0.09,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.2,
            color="white",
            linespacing=1.05,
        )
        ax.text(
            center,
            center_y - 0.065,
            funnel[index]["display_count"],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="white",
        )
        if index in (1, 2):
            ax.text(
                center,
                center_y - height / 2 - 0.055,
                f"{100 * count / raw_count:.1f}% of input",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.2,
                color=PALETTE["neutral_3"],
            )

    arrow_notes = ["40.4%\nkept", "85.4%\nkept", "assigned"]
    for left, right, note in zip(centers[:-1], centers[1:], arrow_notes):
        arrow = FancyArrowPatch(
            (left + 0.108, center_y),
            (right - 0.108, center_y),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color=PALETTE["neutral_3"],
        )
        ax.add_patch(arrow)
        ax.text(
            (left + right) / 2,
            center_y + 0.13,
            note,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.0,
            color=PALETTE["neutral_4"],
            linespacing=1.0,
        )

    split_text = "  ·  ".join(
        f"{row['split'].capitalize()} {int(row['count']):,}" for row in splits
    )
    ax.text(
        0.995,
        0.04,
        split_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=PALETTE["neutral_4"],
    )


def draw_panel_b(ax, sources: list[dict[str, str]]) -> None:
    names = {
        "alexandria": "Alexandria",
        "aflow": "AFLOW",
        "oqmd": "OQMD",
        "wbm": "WBM",
        "mp": "MP",
    }
    labels = [names[row["source"]] for row in sources]
    percentages = np.array([float(row["fraction_pct"]) for row in sources])
    counts = np.array([int(row["count"]) for row in sources])
    total = int(counts.sum())
    colors = [
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["violet"],
        PALETTE["green"],
        PALETTE["gold"],
    ]

    y = np.arange(len(labels))
    bars = ax.barh(y, percentages, height=0.58, color=colors, edgecolor="none", zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 79)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Share of kept records (%)")
    ax.tick_params(axis="y", length=0)
    quiet_axis(ax, grid_axis="x")
    for bar, percentage, count in zip(bars, percentages, counts):
        ax.text(
            percentage + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}%  ·  {compact_count(int(count))}",
            ha="left",
            va="center",
            fontsize=5.7,
            color=PALETTE["neutral_4"],
        )
    title_with_subtitle(
        ax,
        "Final Tier-A source mix",
        f"After tight deduplication; N = {total:,}",
        title_y=1.08,
    )
    panel_label(ax, "b", x=-0.19, y=1.07)


def periodic_table_positions() -> tuple[dict[str, tuple[float, float]], list[tuple[int, int, str]]]:
    rows: list[list[tuple[int, str]]] = [
        [(1, "H"), (18, "He")],
        [
            (1, "Li"),
            (2, "Be"),
            (13, "B"),
            (14, "C"),
            (15, "N"),
            (16, "O"),
            (17, "F"),
            (18, "Ne"),
        ],
        [
            (1, "Na"),
            (2, "Mg"),
            (13, "Al"),
            (14, "Si"),
            (15, "P"),
            (16, "S"),
            (17, "Cl"),
            (18, "Ar"),
        ],
        list(
            zip(
                range(1, 19),
                "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split(),
            )
        ),
        list(
            zip(
                range(1, 19),
                "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split(),
            )
        ),
        [
            (1, "Cs"),
            (2, "Ba"),
            (4, "Hf"),
            (5, "Ta"),
            (6, "W"),
            (7, "Re"),
            (8, "Os"),
            (9, "Ir"),
            (10, "Pt"),
            (11, "Au"),
            (12, "Hg"),
            (13, "Tl"),
            (14, "Pb"),
            (15, "Bi"),
            (16, "Po"),
            (17, "At"),
            (18, "Rn"),
        ],
        [
            (1, "Fr"),
            (2, "Ra"),
            (4, "Rf"),
            (5, "Db"),
            (6, "Sg"),
            (7, "Bh"),
            (8, "Hs"),
            (9, "Mt"),
            (10, "Ds"),
            (11, "Rg"),
            (12, "Cn"),
            (13, "Nh"),
            (14, "Fl"),
            (15, "Mc"),
            (16, "Lv"),
            (17, "Ts"),
            (18, "Og"),
        ],
    ]
    positions: dict[str, tuple[float, float]] = {}
    for period, row in enumerate(rows):
        for group, symbol in row:
            positions[symbol] = (group - 1, float(period))

    lanthanides = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
    actinides = "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
    for offset, symbol in enumerate(lanthanides):
        positions[symbol] = (2 + offset, 7.55)
    for offset, symbol in enumerate(actinides):
        positions[symbol] = (2 + offset, 8.55)
    placeholders = [(3, 6, "57–71"), (3, 7, "89–103")]
    return positions, placeholders


def draw_panel_c(ax, elements: list[dict[str, str]], test_n: int) -> None:
    positions, placeholders = periodic_table_positions()
    element_data = {row["element"]: row for row in elements}
    if set(positions) != set(element_data):
        missing = sorted(set(element_data) - set(positions))
        extra = sorted(set(positions) - set(element_data))
        raise ValueError(f"periodic-table mapping mismatch; missing={missing}, extra={extra}")

    prevalences = np.array([float(row["prevalence_pct"]) for row in elements])
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "figure2_prevalence",
        [PALETTE["neutral_1"], PALETTE["blue_pale"], PALETTE["blue_2"], PALETTE["blue"]],
    )
    norm = mcolors.Normalize(vmin=0, vmax=float(prevalences.max()))
    cell_width = 0.90
    cell_height = 0.78

    for symbol, (x, y) in positions.items():
        row = element_data[symbol]
        prevalence = float(row["prevalence_pct"])
        atomic_number = int(row["atomic_number"])
        face = cmap(norm(prevalence)) if prevalence > 0 else PALETTE["neutral_1"]
        edge = "white" if prevalence > 0 else PALETTE["neutral_2"]
        ax.add_patch(
            Rectangle(
                (x + 0.05, y + 0.04),
                cell_width,
                cell_height,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.45,
            )
        )
        text_color = "white" if norm(prevalence) > 0.55 else PALETTE["black"]
        ax.text(
            x + 0.12,
            y + 0.15,
            str(atomic_number),
            ha="left",
            va="top",
            fontsize=3.1,
            color=text_color,
        )
        ax.text(
            x + 0.50,
            y + 0.36,
            symbol,
            ha="center",
            va="center",
            fontsize=5.7,
            fontweight="bold",
            color=text_color,
        )
        value_label = f"{prevalence:.1f}" if prevalence > 0 else "—"
        ax.text(
            x + 0.50,
            y + 0.67,
            value_label,
            ha="center",
            va="center",
            fontsize=3.25,
            color=text_color,
        )

    for group, period, label in placeholders:
        x, y = group - 1, period - 1
        ax.add_patch(
            Rectangle(
                (x + 0.05, y + 0.04),
                cell_width,
                cell_height,
                facecolor=PALETTE["neutral_1"],
                edgecolor=PALETTE["neutral_2"],
                linewidth=0.45,
            )
        )
        ax.text(
            x + 0.50,
            y + 0.43,
            label,
            ha="center",
            va="center",
            fontsize=3.6,
            color=PALETTE["neutral_3"],
        )

    ax.text(
        0.1,
        7.97,
        "Lanthanides",
        ha="left",
        va="center",
        fontsize=4.7,
        color=PALETTE["neutral_3"],
    )
    ax.text(
        0.1,
        8.97,
        "Actinides",
        ha="left",
        va="center",
        fontsize=4.7,
        color=PALETTE["neutral_3"],
    )
    ax.set_xlim(-0.05, 18.02)
    ax.set_ylim(9.45, -0.12)
    ax.set_axis_off()

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_ax = ax.inset_axes([0.20, 0.91, 0.34, 0.027])
    colorbar = plt.colorbar(scalar_mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.outline.set_linewidth(0.45)
    colorbar.outline.set_edgecolor(PALETTE["neutral_3"])
    colorbar.ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)
    colorbar.set_label("Record prevalence (%)", fontsize=5.2, labelpad=1.5)
    colorbar.set_ticks([0, 5, 10])

    title_with_subtitle(
        ax,
        "Element prevalence across the periodic table",
        (
            f"Complete test set; N = {test_n:,} · percentage of records containing "
            "each element (one count per element per record)"
        ),
        title_y=1.045,
    )
    panel_label(ax, "c", x=-0.035, y=1.04)


def draw_panel_d(
    ax,
    n_sites_rows: list[dict[str, str]],
    summary: dict[str, Any],
) -> None:
    x = np.array([int(row["n_sites"]) for row in n_sites_rows])
    counts = np.array([int(row["record_count"]) for row in n_sites_rows])
    test_n = int(summary["record_count"])
    median = float(summary["n_sites"]["median"])
    p95 = float(summary["n_sites"]["p95"])

    ax.bar(
        x,
        counts,
        width=0.90,
        color=PALETTE["blue_pale"],
        edgecolor=PALETTE["blue_2"],
        linewidth=0.25,
        zorder=2,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 129)
    ax.set_ylim(0.7, 10_000)
    ax.set_xticks([1, 32, 64, 96, 128])
    ax.set_yticks([1, 10, 100, 1_000, 10_000])
    ax.set_xlabel(r"Sites per unit cell, $n_{\mathrm{sites}}$")
    ax.set_ylabel("Records (log scale)")
    quiet_axis(ax, grid_axis="y")
    ax.axvline(
        median,
        color=PALETTE["red"],
        linewidth=0.9,
        linestyle="--",
        zorder=3,
        label=f"Median {median:g}",
    )
    ax.axvline(
        p95,
        color=PALETTE["neutral_4"],
        linewidth=0.9,
        linestyle=":",
        zorder=3,
        label=f"95th pct. {p95:g}",
    )
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.8)
    title_with_subtitle(
        ax,
        "Structure size",
        f"Test N = {test_n:,} · 1–128 sites",
        title_y=1.12,
    )
    panel_label(ax, "d", x=-0.24, y=1.11)


def draw_panel_e(ax, systems: list[dict[str, str]], test_n: int) -> None:
    labels = [row["crystal_system"].capitalize() for row in systems]
    percentages = np.array([float(row["fraction_pct"]) for row in systems])
    y = np.arange(len(systems))
    colors = [
        PALETTE["neutral_3"],
        PALETTE["teal"],
        PALETTE["blue_2"],
        PALETTE["blue"],
        PALETTE["violet"],
        PALETTE["green"],
        PALETTE["gold"],
    ]
    bars = ax.barh(y, percentages, height=0.58, color=colors, edgecolor="none", zorder=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 29)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel("Records (%)")
    ax.tick_params(axis="y", length=0, labelsize=5.5)
    quiet_axis(ax, grid_axis="x")
    for bar, percentage in zip(bars, percentages):
        ax.text(
            percentage + 0.55,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}",
            ha="left",
            va="center",
            fontsize=5.4,
            color=PALETTE["neutral_4"],
        )
    title_with_subtitle(
        ax,
        "Crystal systems",
        f"Test N = {test_n:,} · standard SG mapping",
        title_y=1.12,
    )
    panel_label(ax, "e", x=-0.12, y=1.11)


def gaussian_trace(x: np.ndarray, peaks: list[tuple[float, float]], width: float) -> np.ndarray:
    y = np.zeros_like(x)
    for position, intensity in peaks:
        y += intensity * np.exp(-0.5 * ((x - position) / width) ** 2)
    return y / y.max()


def draw_panel_f(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title_with_subtitle(
        ax,
        "Multimodal record (schematic)",
        "Structure-derived signals and serialized target",
        title_y=1.12,
    )
    panel_label(ax, "f", x=-0.10, y=1.11)

    structure_box = FancyBboxPatch(
        (0.01, 0.30),
        0.26,
        0.44,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=PALETTE["neutral_1"],
        edgecolor=PALETTE["neutral_2"],
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(structure_box)
    cell_front = np.array([[0.055, 0.43], [0.155, 0.39], [0.155, 0.55], [0.055, 0.59]])
    cell_back = cell_front + np.array([0.055, 0.075])
    for cell in (cell_front, cell_back):
        ax.plot(
            np.r_[cell[:, 0], cell[0, 0]],
            np.r_[cell[:, 1], cell[0, 1]],
            color=PALETTE["neutral_4"],
            lw=0.75,
            transform=ax.transAxes,
        )
    for front, back in zip(cell_front, cell_back):
        ax.plot(
            [front[0], back[0]],
            [front[1], back[1]],
            color=PALETTE["neutral_4"],
            lw=0.75,
            transform=ax.transAxes,
        )
    atoms = [
        (0.060, 0.435, PALETTE["blue"], 0.015),
        (0.155, 0.550, PALETTE["teal"], 0.013),
        (0.210, 0.625, PALETTE["violet"], 0.015),
        (0.125, 0.520, PALETTE["gold"], 0.012),
    ]
    for x_atom, y_atom, color, radius in atoms:
        ax.add_patch(
            Circle(
                (x_atom, y_atom),
                radius,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )
        )
    ax.text(
        0.14,
        0.345,
        "Crystal structure",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )

    box_specs = [
        (0.43, 0.67, 0.55, 0.23, PALETTE["teal_pale"]),
        (0.43, 0.40, 0.55, 0.19, PALETTE["violet_pale"]),
        (0.43, 0.08, 0.55, 0.24, PALETTE["blue_pale"]),
    ]
    for x_box, y_box, width, height, face in box_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x_box, y_box),
                width,
                height,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                facecolor=face,
                edgecolor="none",
                transform=ax.transAxes,
            )
        )

    branch_x = 0.355
    ax.plot(
        [0.27, branch_x],
        [0.52, 0.52],
        transform=ax.transAxes,
        color=PALETTE["neutral_3"],
        linewidth=0.8,
    )
    ax.plot(
        [branch_x, branch_x],
        [0.20, 0.79],
        transform=ax.transAxes,
        color=PALETTE["neutral_3"],
        linewidth=0.8,
    )
    for destination_y in (0.79, 0.495, 0.20):
        ax.add_patch(
            FancyArrowPatch(
                (branch_x, destination_y),
                (0.425, destination_y),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.8,
                color=PALETTE["neutral_3"],
            )
        )

    ax.text(
        0.455,
        0.865,
        "PXRD / ND",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )
    ax.text(
        0.955,
        0.865,
        r"dense $I(2\theta)$ vectors",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=4.8,
        color=PALETTE["neutral_3"],
    )
    spectrum_x = np.linspace(0, 1, 240)
    pxrd = gaussian_trace(
        spectrum_x,
        [(0.15, 0.45), (0.32, 1.0), (0.58, 0.6), (0.79, 0.38)],
        0.018,
    )
    neutron = gaussian_trace(
        spectrum_x,
        [(0.15, 0.25), (0.32, 0.68), (0.58, 1.0), (0.79, 0.55)],
        0.018,
    )
    x_plot = 0.50 + 0.44 * spectrum_x
    ax.plot(
        x_plot,
        0.775 + 0.055 * pxrd,
        transform=ax.transAxes,
        color=PALETTE["neutral_3"],
        lw=0.8,
    )
    ax.plot(
        x_plot,
        0.695 + 0.055 * neutron,
        transform=ax.transAxes,
        color=PALETTE["teal"],
        lw=0.8,
    )
    ax.text(0.465, 0.795, "X-ray", transform=ax.transAxes, fontsize=4.5, va="center")
    ax.text(0.465, 0.715, "Neutron", transform=ax.transAxes, fontsize=4.5, va="center")

    ax.text(
        0.455,
        0.545,
        "XRF element mass fractions",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )
    ax.text(
        0.455,
        0.435,
        "structured values — not a raw energy spectrum",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.6,
        color=PALETTE["neutral_3"],
    )
    for y_bar, width, color, symbol in [
        (0.505, 0.16, PALETTE["violet"], "A"),
        (0.475, 0.10, PALETTE["teal"], "B"),
    ]:
        ax.text(
            0.72,
            y_bar,
            symbol,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=4.5,
        )
        ax.add_patch(
            Rectangle(
                (0.735, y_bar - 0.009),
                width,
                0.018,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
            )
        )

    ax.text(
        0.455,
        0.268,
        "Crystal target tokens",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.8,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )
    ax.text(
        0.455,
        0.155,
        "<FORMULA> … <SG###> …\n<LATTICE> … <ATOM> …",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.1,
        family="monospace",
        linespacing=1.35,
        color=PALETTE["blue"],
    )


def main() -> None:
    summary_path = DATA_DIR / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"prepared data missing: {summary_path}; run prepare_data.py first"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    funnel = read_csv("data_governance_funnel.csv")
    sources = read_csv("source_composition.csv")
    splits = read_csv("split_sizes.csv")
    elements = read_csv("element_prevalence.csv")
    n_sites = read_csv("n_sites_distribution.csv")
    systems = read_csv("crystal_system_distribution.csv")

    test_n = int(summary["test_statistics"]["record_count"])
    if test_n != 33_944:
        raise ValueError(f"Figure 2 requires the frozen test set N=33,944; got {test_n:,}")
    if sum(int(row["record_count"]) for row in systems) != test_n:
        raise ValueError("crystal-system data do not sum to the test-set size")
    if sum(int(row["record_count"]) for row in n_sites) != test_n:
        raise ValueError("n_sites data do not sum to the test-set size")

    apply_style()
    # Keep the shared tight export while preserving the requested ~7.2-inch width.
    plt.rcParams["savefig.pad_inches"] = 0.03
    fig = plt.figure(figsize=(7.2, 7.9))
    grid = fig.add_gridspec(
        3,
        14,
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.96,
        hspace=0.54,
        wspace=1.35,
        height_ratios=[1.45, 2.55, 2.05],
    )
    ax_a = fig.add_subplot(grid[0, :8])
    ax_b = fig.add_subplot(grid[0, 8:])
    ax_c = fig.add_subplot(grid[1, :])
    ax_d = fig.add_subplot(grid[2, :4])
    ax_e = fig.add_subplot(grid[2, 4:8])
    ax_f = fig.add_subplot(grid[2, 8:])
    # Give panel e's long crystal-system tick labels a clean gutter.
    position_e = ax_e.get_position()
    panel_e_shift = 0.036
    ax_e.set_position(
        [
            position_e.x0 + panel_e_shift,
            position_e.y0,
            position_e.width - panel_e_shift,
            position_e.height,
        ]
    )

    draw_panel_a(ax_a, funnel, splits)
    draw_panel_b(ax_b, sources)
    draw_panel_c(ax_c, elements, test_n)
    draw_panel_d(ax_d, n_sites, summary["test_statistics"])
    draw_panel_e(ax_e, systems, test_n)
    draw_panel_f(ax_f)

    source_note(
        fig,
        (
            "Sources: metadata_index_summary.json (a,b); export_summary.json (a); "
            "complete test.jsonl scan (c–e; N = 33,944; no sampling). Panel f is schematic."
        ),
        y=0.012,
    )
    save_figure(fig, OUTPUT_DIR, "figure_2_dataset_atlas", dpi=300)
    plt.close(fig)
    print(
        "Saved Figure 2 to "
        + ", ".join(
            str(OUTPUT_DIR / f"figure_2_dataset_atlas.{extension}")
            for extension in ("png", "pdf", "svg")
        )
    )


if __name__ == "__main__":
    main()
