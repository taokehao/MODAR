#!/usr/bin/env python3
"""Render manuscript Figure 3 from its curated rerank-only source-data CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.transforms import Bbox


FIGURE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(FIGURE_DIR.parent))

from common.style import (  # noqa: E402
    CONDITION_COLORS,
    PALETTE,
    apply_style,
    panel_label,
    quiet_axis,
    source_note,
)


DATA_CSV = FIGURE_DIR / "data" / "figure_3_source_data.csv"
OUTPUT_DIR = FIGURE_DIR / "outputs"
OUTPUT_STEM = "figure_3_multimodal_performance"

CONDITIONS = (
    "pxrd",
    "pxrd_neutron",
    "pxrd_xrf",
    "pxrd_neutron_xrf",
    "pxrd_formula",
)
CONDITION_LABELS = {
    "pxrd": "PXRD",
    "pxrd_neutron": "PXRD + ND",
    "pxrd_xrf": "PXRD + XRF",
    "pxrd_neutron_xrf": "PXRD + ND + XRF",
    "pxrd_formula": "Formula reference",
}
SHORT_LABELS = {
    "pxrd": "PXRD",
    "pxrd_neutron": "+ ND",
    "pxrd_xrf": "+ XRF",
    "pxrd_neutron_xrf": "Multi",
    "pxrd_formula": "Formula ref.",
}
BAR_LABELS = ("PXRD", "+ND", "+XRF", "Multi", "Formula\nref.")
LATTICES = ("lattice_free", "known_lattice")
TOP_K = (1, 5, 10, 20, 50)


def load_data() -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load and validate the 2 × 5 × 5 plotting table."""

    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    with DATA_CSV.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["rank"] = int(row["rank"])
            row["n_records"] = int(row["n_records"])
            row["n_candidates"] = int(row["n_candidates"])
            row["composition_rate_percent"] = float(
                row["composition_rate_percent"]
            )
            row["success_rate_percent"] = float(row["success_rate_percent"])

            key = (row["lattice_key"], row["condition_key"], row["rank"])
            if key in lookup:
                raise ValueError(f"Duplicate data row: {key}")
            lookup[key] = row

    expected = {
        (lattice, condition, rank)
        for lattice in LATTICES
        for condition in CONDITIONS
        for rank in TOP_K
    }
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"Unexpected CSV keys; missing={missing}, extra={extra}")

    if {row["metric"] for row in lookup.values()} != {"topk_rerank"}:
        raise ValueError("Figure 3 must use rerank metrics only")
    if {row["n_records"] for row in lookup.values()} != {33_944}:
        raise ValueError("Figure 3 requires the full test N=33,944")
    if {row["n_candidates"] for row in lookup.values()} != {100}:
        raise ValueError("Figure 3 requires N=100 candidates/sample")
    return lookup


def value(
    data: dict[tuple[str, str, int], dict[str, Any]],
    lattice: str,
    condition: str,
    rank: int,
) -> float:
    return float(data[(lattice, condition, rank)]["success_rate_percent"])


def composition(
    data: dict[tuple[str, str, int], dict[str, Any]],
    lattice: str,
    condition: str,
) -> float:
    return float(data[(lattice, condition, 1)]["composition_rate_percent"])


def draw_information_ladder(ax: plt.Axes) -> None:
    """Panel a: modality branches merge into the multimodal condition."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Information ladder", loc="left", pad=5)
    panel_label(ax, "a", x=-0.10, y=1.03)

    def node(
        center: tuple[float, float],
        size: tuple[float, float],
        text: str,
        *,
        facecolor: str,
        edgecolor: str,
        fontsize: float = 7.0,
    ) -> None:
        width, height = size
        x = center[0] - width / 2
        y = center[1] - height / 2
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.text(
            center[0],
            center[1],
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=PALETTE["black"],
            linespacing=0.95,
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.8,
                color=PALETTE["neutral_3"],
                shrinkA=1.5,
                shrinkB=1.5,
            )
        )

    node(
        (0.10, 0.50),
        (0.19, 0.18),
        "PXRD",
        facecolor=PALETTE["neutral_1"],
        edgecolor=PALETTE["neutral_3"],
    )
    node(
        (0.34, 0.72),
        (0.19, 0.17),
        "+ ND",
        facecolor=PALETTE["teal_pale"],
        edgecolor=PALETTE["teal"],
    )
    node(
        (0.34, 0.28),
        (0.19, 0.17),
        "+ XRF",
        facecolor=PALETTE["violet_pale"],
        edgecolor=PALETTE["violet"],
    )
    node(
        (0.62, 0.50),
        (0.22, 0.23),
        "Multi\nPXRD + ND\n+ XRF",
        facecolor=PALETTE["blue_pale"],
        edgecolor=PALETTE["blue"],
        fontsize=5.8,
    )
    node(
        (0.90, 0.50),
        (0.18, 0.22),
        "Formula\nreference",
        facecolor=PALETTE["neutral_1"],
        edgecolor=PALETTE["neutral_4"],
        fontsize=6.4,
    )

    arrow((0.20, 0.53), (0.245, 0.67))
    arrow((0.20, 0.47), (0.245, 0.33))
    arrow((0.44, 0.68), (0.505, 0.56))
    arrow((0.44, 0.32), (0.505, 0.44))
    arrow((0.735, 0.50), (0.805, 0.50))

    ax.text(
        0.34,
        0.91,
        "structural contrast",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        color=PALETTE["neutral_3"],
    )
    ax.text(
        0.34,
        0.09,
        "elemental constraint",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        color=PALETTE["neutral_3"],
    )


def draw_core_bars(
    ax: plt.Axes,
    data: dict[tuple[str, str, int], dict[str, Any]],
    lattice: str,
    title: str,
    label: str,
    *,
    show_legend: bool,
) -> None:
    """Panels b/c: paired Top-1 and Top-20 bars for all five conditions."""

    x = np.arange(len(CONDITIONS))
    width = 0.31
    colors = [CONDITION_COLORS[CONDITION_LABELS[key]] for key in CONDITIONS]
    top1 = [value(data, lattice, key, 1) for key in CONDITIONS]
    top20 = [value(data, lattice, key, 20) for key in CONDITIONS]

    bars_1 = ax.bar(
        x - width / 2,
        top1,
        width,
        color=colors,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    bars_20 = ax.bar(
        x + width / 2,
        top20,
        width,
        color=[mpl.colors.to_rgba(color, 0.20) for color in colors],
        edgecolor=colors,
        linewidth=0.75,
        zorder=3,
    )

    for bars, values in ((bars_1, top1), (bars_20, top20)):
        for bar, metric_value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                metric_value + 1.0,
                f"{metric_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=PALETTE["neutral_4"],
            )

    ax.set_title(title, loc="left", pad=5)
    panel_label(ax, label, x=-0.13, y=1.03)
    ax.set_ylim(0, 104)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x, BAR_LABELS)
    ax.tick_params(axis="x", labelsize=6.0, pad=2)
    ax.margins(x=0.025)
    quiet_axis(ax, grid_axis="y")

    if show_legend:
        top1_handle = Patch(
            facecolor=PALETTE["neutral_4"],
            edgecolor="none",
            label="Top-1",
        )
        top20_handle = Patch(
            facecolor=mpl.colors.to_rgba(PALETTE["neutral_4"], 0.16),
            edgecolor=PALETTE["neutral_4"],
            linewidth=0.7,
            label="Top-20",
        )
        ax.legend(
            handles=(top1_handle, top20_handle),
            loc="upper left",
            ncol=2,
            handlelength=1.1,
            columnspacing=0.8,
            borderaxespad=0.2,
            fontsize=6.0,
        )


def draw_composition_relation(
    ax: plt.Axes,
    data: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    """Panel d: composition recovery against reranked Top-1 success."""

    for condition in CONDITIONS:
        color = CONDITION_COLORS[CONDITION_LABELS[condition]]
        x_values = [composition(data, lattice, condition) for lattice in LATTICES]
        y_values = [value(data, lattice, condition, 1) for lattice in LATTICES]
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=0.75,
            alpha=0.48,
            zorder=2,
        )
        ax.scatter(
            x_values[0],
            y_values[0],
            s=23,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=4,
        )
        ax.scatter(
            x_values[1],
            y_values[1],
            s=22,
            marker="s",
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            zorder=4,
        )

    annotation_specs = {
        "pxrd": ("known_lattice", (5, 6)),
        "pxrd_neutron": ("known_lattice", (5, 3)),
        "pxrd_xrf": ("lattice_free", (-48, 5)),
        "pxrd_neutron_xrf": ("known_lattice", (-44, 3)),
        "pxrd_formula": ("lattice_free", (-50, -33)),
    }
    for condition, (lattice, offset) in annotation_specs.items():
        color = CONDITION_COLORS[CONDITION_LABELS[condition]]
        xy = (
            composition(data, lattice, condition),
            value(data, lattice, condition, 1),
        )
        ax.annotate(
            SHORT_LABELS[condition],
            xy=xy,
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.0,
            color=color,
            fontweight="bold" if condition == "pxrd_neutron_xrf" else "normal",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.55,
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )

    marker_legend = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=PALETTE["neutral_4"],
            markersize=4,
            label="Lattice-free",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=PALETTE["neutral_4"],
            markeredgecolor=PALETTE["neutral_4"],
            markersize=3.8,
            label="Known lattice",
        ),
    )
    ax.legend(
        handles=marker_legend,
        loc="center",
        bbox_to_anchor=(0.56, 0.37),
        fontsize=5.8,
        handletextpad=0.3,
    )
    ax.set_title("Composition recovery vs Top-1", loc="left", pad=5)
    panel_label(ax, "d", x=-0.13, y=1.03)
    ax.set_xlim(20, 102)
    ax.set_ylim(0, 100)
    ax.set_xticks((20, 40, 60, 80, 100))
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_xlabel("Composition recovery (%)")
    ax.set_ylabel("Reranked Top-1 (%)")
    quiet_axis(ax, grid_axis="both")


def draw_topk_facets(
    fig: plt.Figure,
    spec: mpl.gridspec.SubplotSpec,
    data: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    """Panel e: Top-k curves faceted by lattice information."""

    subgrid = spec.subgridspec(1, 2, wspace=0.13)
    axes = [fig.add_subplot(subgrid[0, i]) for i in range(2)]
    positions = np.arange(len(TOP_K))
    multi_color = CONDITION_COLORS["PXRD + ND + XRF"]
    reference_color = CONDITION_COLORS["Formula reference"]

    for index, (ax, lattice, title) in enumerate(
        zip(axes, LATTICES, ("Lattice-free", "Known lattice"))
    ):
        multi_values = [
            value(data, lattice, "pxrd_neutron_xrf", rank) for rank in TOP_K
        ]
        reference_values = [
            value(data, lattice, "pxrd_formula", rank) for rank in TOP_K
        ]
        ax.plot(
            positions,
            multi_values,
            color=multi_color,
            marker="o",
            markersize=2.8,
            linewidth=1.25,
            label="Multi",
            zorder=3,
        )
        ax.plot(
            positions,
            reference_values,
            color=reference_color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.7,
            markersize=2.8,
            linewidth=1.05,
            label="Formula ref.",
            zorder=3,
        )
        ax.set_title(title, fontsize=7.2, pad=4)
        ax.set_ylim(82, 96)
        ax.set_yticks((84, 88, 92, 96))
        ax.set_xticks(positions, [str(rank) for rank in TOP_K])
        ax.tick_params(axis="x", labelsize=5.8, pad=2)
        ax.set_xlabel("Rank k", fontsize=6.5)
        quiet_axis(ax, grid_axis="y")
        if index == 0:
            ax.set_ylabel("Success (%)", fontsize=6.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)

    panel_label(axes[0], "e", x=-0.34, y=1.04)
    axes[0].text(
        0.00,
        1.14,
        "Top-k retrieval",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color=PALETTE["black"],
    )
    axes[1].legend(
        loc="lower right",
        fontsize=5.4,
        handlelength=1.4,
        handletextpad=0.3,
        borderaxespad=0.25,
    )


def draw_reference_gap(
    ax: plt.Axes,
    data: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    """Panel f: multimodal distance to the formula reference and key gain."""

    comparisons = (
        ("Free · T1", "lattice_free", 1),
        ("Free · T20", "lattice_free", 20),
        ("Known · T1", "known_lattice", 1),
        ("Known · T20", "known_lattice", 20),
    )
    y_positions = np.array((2.7, 1.9, 1.1, 0.3))
    deltas: list[float] = []
    pairs: list[tuple[float, float]] = []
    for _, lattice, rank in comparisons:
        multi = value(data, lattice, "pxrd_neutron_xrf", rank)
        reference = value(data, lattice, "pxrd_formula", rank)
        deltas.append(multi - reference)
        pairs.append((multi, reference))

    colors = (
        PALETTE["blue_2"],
        mpl.colors.to_rgba(PALETTE["blue_2"], 0.45),
        PALETTE["blue"],
        mpl.colors.to_rgba(PALETTE["blue"], 0.45),
    )
    edgecolors = (
        PALETTE["blue_2"],
        PALETTE["blue_2"],
        PALETTE["blue"],
        PALETTE["blue"],
    )
    bars = ax.barh(
        y_positions,
        deltas,
        height=0.45,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.65,
        zorder=3,
    )
    ax.axvline(0, color=PALETTE["neutral_4"], linewidth=0.8, zorder=2)

    for bar, delta, (multi, reference), y in zip(
        bars, deltas, pairs, y_positions
    ):
        sign = "+" if delta >= 0 else "−"
        delta_text = f"{sign}{abs(delta):.1f}"
        if delta >= 0.4:
            delta_x = delta / 2
            delta_ha = "center"
            delta_color = "white"
        else:
            delta_x = delta - 0.08
            delta_ha = "right"
            delta_color = PALETTE["neutral_4"]
        ax.text(
            delta_x,
            bar.get_y() + bar.get_height() / 2,
            delta_text,
            ha=delta_ha,
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color=delta_color,
        )
        ax.text(
            2.22,
            y,
            f"{multi:.1f} vs {reference:.1f}",
            ha="right",
            va="center",
            fontsize=5.6,
            color=PALETTE["neutral_4"],
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    free_gain = value(data, "lattice_free", "pxrd_neutron_xrf", 1) - value(
        data, "lattice_free", "pxrd", 1
    )
    known_gain = value(data, "known_lattice", "pxrd_neutron_xrf", 1) - value(
        data, "known_lattice", "pxrd", 1
    )
    ax.text(
        -0.10,
        3.55,
        "Top-1 gain over PXRD\n"
        f"+{free_gain:.1f} pp free  |  +{known_gain:.1f} pp known",
        ha="center",
        va="center",
        fontsize=5.5,
        color=PALETTE["blue"],
        linespacing=1.15,
    )
    ax.text(
        2.22,
        3.25,
        "Multi vs ref.",
        ha="right",
        va="center",
        fontsize=5.4,
        color=PALETTE["neutral_3"],
    )

    ax.set_title("Formula-reference gap", loc="left", pad=5)
    panel_label(ax, "f", x=-0.13, y=1.03)
    ax.set_xlim(-2.6, 2.3)
    ax.set_ylim(-0.2, 3.9)
    ax.set_yticks(y_positions, [item[0] for item in comparisons])
    ax.tick_params(axis="y", labelsize=6.2)
    ax.set_xticks((-2, -1, 0, 1, 2))
    ax.set_xlabel("Multi − Formula reference\n(percentage points)")
    quiet_axis(ax, grid_axis="x")


def main() -> None:
    apply_style()
    data = load_data()

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.070,
        right=0.985,
        bottom=0.105,
        top=0.950,
        width_ratios=(1.02, 1.00, 1.08),
        height_ratios=(0.96, 1.04),
        wspace=0.46,
        hspace=0.48,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_f = fig.add_subplot(grid[1, 2])

    draw_information_ladder(ax_a)
    draw_core_bars(
        ax_b,
        data,
        "lattice_free",
        "Lattice-free retrieval",
        "b",
        show_legend=True,
    )
    draw_core_bars(
        ax_c,
        data,
        "known_lattice",
        "Known-lattice retrieval",
        "c",
        show_legend=False,
    )
    draw_composition_relation(ax_d, data)
    draw_topk_facets(fig, grid[1, 1], data)
    draw_reference_gap(ax_f, data)

    source_note(
        fig,
        "Source: ladder_test_table.json; full test N=33,944; "
        "N=100 candidates/sample; rerank metrics.\nFormula reference is an "
        "oracle reference, not a strict theoretical upper bound.",
        y=0.008,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_figure_bbox = Bbox.from_bounds(0, 0, *fig.get_size_inches())
    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_STEM}.png",
        dpi=300,
        bbox_inches=full_figure_bbox,
        pad_inches=0,
    )
    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_STEM}.pdf",
        bbox_inches=full_figure_bbox,
        pad_inches=0,
    )
    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_STEM}.svg",
        bbox_inches=full_figure_bbox,
        pad_inches=0,
    )
    plt.close(fig)

    print(
        f"Saved {OUTPUT_STEM}.png/.pdf/.svg to "
        f"{OUTPUT_DIR.relative_to(FIGURE_DIR.parents[1])}"
    )


if __name__ == "__main__":
    main()
