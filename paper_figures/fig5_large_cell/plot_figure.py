#!/usr/bin/env python3
"""Render manuscript Figure 5 from its curated CSV source data."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_DIR = FIGURE_DIR / "outputs"

sys.path.insert(0, str(REPO_ROOT))
from paper_figures.common.style import (  # noqa: E402
    PALETTE,
    add_bar_labels,
    apply_style,
    panel_label,
    quiet_axis,
    save_figure,
    source_note,
)


FULL_DATA = DATA_DIR / "full_test_size_bins.csv"
CAPACITY_DATA = DATA_DIR / "train_capacity.csv"
SUBSET_DATA = DATA_DIR / "bigcell_same_subset.csv"
GAIN_DATA = DATA_DIR / "bigcell_gains.csv"

METHOD_ORDER = (
    "Base multimodal",
    "Large-cell specialist",
    "Base formula reference",
)
METHOD_COLORS = {
    "Base multimodal": PALETTE["blue_2"],
    "Large-cell specialist": PALETTE["blue"],
    "Base formula reference": PALETTE["neutral_4"],
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(FIGURE_DIR)}; run prepare_data.py first"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _value(
    rows: list[dict[str, str]],
    *,
    value_column: str,
    **criteria: object,
) -> float:
    matches = [
        row
        for row in rows
        if all(row[key] == str(expected) for key, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(matches)}")
    return float(matches[0][value_column])


def _plot_full_test_panel(
    ax,
    rows: list[dict[str, str]],
    *,
    top_k: int,
    label: str,
) -> None:
    groups = (
        ("Lattice-free", "<=20", "≤20\nfree"),
        ("Lattice-free", ">20", ">20\nfree"),
        ("Known lattice", "<=20", "≤20\nknown"),
        ("Known lattice", ">20", ">20\nknown"),
    )
    methods = ("Base multimodal", "Base formula reference")
    colors = (METHOD_COLORS[methods[0]], METHOD_COLORS[methods[1]])
    markers = ("o", "D")
    x = np.arange(len(groups), dtype=float)
    offsets = (-0.08, 0.08)

    for group_x, (lattice, size_bin, _) in zip(x, groups, strict=True):
        values = [
            _value(
                rows,
                value_column="match_rate_percent",
                lattice=lattice,
                size_bin=size_bin,
                method=method,
                top_k=top_k,
            )
            for method in methods
        ]
        ax.plot(
            [group_x + offsets[0], group_x + offsets[1]],
            values,
            color=PALETTE["neutral_2"],
            lw=1.2,
            zorder=1,
        )
        for method_index, (value, color, marker) in enumerate(
            zip(values, colors, markers, strict=True)
        ):
            point_x = group_x + offsets[method_index]
            ax.scatter(
                point_x,
                value,
                s=25,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=methods[method_index] if group_x == 0 else None,
            )
            ax.annotate(
                f"{value:.1f}",
                (point_x, value),
                xytext=(0, -8 if method_index == 0 else 4),
                textcoords="offset points",
                ha="center",
                va="top" if method_index == 0 else "bottom",
                fontsize=5.8,
                color=color,
            )

    ax.axvline(1.5, color=PALETTE["neutral_2"], lw=0.7, ls=(0, (2, 2)))
    ax.set_xticks(x, [group[2] for group in groups])
    ax.set_ylim(48, 101.5)
    ax.set_yticks((50, 60, 70, 80, 90, 100))
    ax.set_ylabel("Structure match rate (%)")
    ax.set_title(
        f"Full test: Top-{top_k} (N=33,944)",
        loc="left",
        pad=7,
        fontsize=8.4,
    )
    ax.text(
        0.5,
        -0.27,
        "bin n = 30,159 / 3,785",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color=PALETTE["neutral_3"],
    )
    quiet_axis(ax, grid_axis="y")
    panel_label(ax, label, x=-0.16, y=1.05)
    if top_k == 1:
        ax.legend(
            loc="center",
            bbox_to_anchor=(0.5, 0.53),
            ncol=2,
            handletextpad=0.35,
            columnspacing=0.8,
            borderaxespad=0,
            fontsize=5.8,
        )


def _rounded_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    text: str,
    text_color: str = PALETTE["black"],
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=text_color,
        linespacing=1.25,
    )


def _plot_capacity_panel(
    ax,
    rows: list[dict[str, str]],
    *,
    label: str,
) -> None:
    capacity = {
        row["size_bin"]: {
            "n": int(row["n_train"]),
            "share": float(row["share_percent"]),
        }
        for row in rows
    }
    small_share = capacity["<=20"]["share"] / 100.0
    total_train = capacity["<=20"]["n"] + capacity[">20"]["n"]

    ax.set_axis_off()
    ax.set_title(
        "Training imbalance → routing",
        loc="left",
        pad=7,
        fontsize=8.4,
    )
    panel_label(ax, label, x=-0.08, y=1.05)

    bar_x, bar_y, bar_w, bar_h = 0.04, 0.75, 0.92, 0.10
    ax.text(
        bar_x,
        0.89,
        f"Base training mix (N = {total_train:,})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["neutral_4"],
    )
    ax.add_patch(
        FancyBboxPatch(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            boxstyle="round,pad=0,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=PALETTE["neutral_1"],
            edgecolor=PALETTE["neutral_3"],
            linewidth=0.7,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (bar_x, bar_y),
            bar_w * small_share,
            bar_h,
            transform=ax.transAxes,
            facecolor=PALETTE["blue_pale"],
            edgecolor="none",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (bar_x + bar_w * small_share, bar_y),
            bar_w * (1.0 - small_share),
            bar_h,
            transform=ax.transAxes,
            facecolor=PALETTE["red"],
            edgecolor="none",
        )
    )
    ax.text(
        bar_x + 0.02,
        bar_y + bar_h / 2,
        f"≤20  {capacity['<=20']['n']:,} ({capacity['<=20']['share']:.1f}%)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.5,
        color=PALETTE["blue"],
    )
    ax.annotate(
        f">20  {capacity['>20']['n']:,}\n({capacity['>20']['share']:.1f}%)",
        xy=(bar_x + bar_w * (small_share + (1.0 - small_share) / 2), bar_y),
        xycoords=ax.transAxes,
        xytext=(0.95, 0.64),
        textcoords=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=PALETTE["red"],
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["red"],
            "lw": 0.7,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )

    _rounded_box(
        ax,
        (0.04, 0.28),
        0.25,
        0.18,
        facecolor=PALETTE["neutral_1"],
        edgecolor=PALETTE["neutral_3"],
        text="test crystal\nsite count",
    )
    _rounded_box(
        ax,
        (0.64, 0.43),
        0.31,
        0.16,
        facecolor=PALETTE["blue_pale"],
        edgecolor=PALETTE["blue_2"],
        text="≤20\nbase model",
        text_color=PALETTE["blue"],
    )
    _rounded_box(
        ax,
        (0.64, 0.13),
        0.31,
        0.16,
        facecolor=PALETTE["red_pale"],
        edgecolor=PALETTE["red"],
        text=">20\nlarge-cell specialist",
        text_color=PALETTE["red"],
    )
    for end_y, color in ((0.51, PALETTE["blue_2"]), (0.21, PALETTE["red"])):
        ax.annotate(
            "",
            xy=(0.64, end_y),
            xycoords=ax.transAxes,
            xytext=(0.29, 0.37),
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.1,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )
    ax.text(
        0.46,
        0.38,
        "route by\ncell size",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=PALETTE["neutral_3"],
    )


def _plot_subset_bars(
    ax,
    rows: list[dict[str, str]],
    *,
    lattice: str,
    label: str,
    show_legend: bool,
) -> None:
    top_ks = (1, 20)
    x = np.arange(len(top_ks), dtype=float)
    width = 0.23
    offsets = (-width, 0.0, width)
    legend_labels = {
        "Base multimodal": "Base multi.",
        "Large-cell specialist": "Specialist multi.",
        "Base formula reference": "Base formula ref.",
    }

    for offset, method in zip(offsets, METHOD_ORDER, strict=True):
        values = [
            _value(
                rows,
                value_column="match_rate_percent",
                lattice=lattice,
                method=method,
                top_k=top_k,
            )
            for top_k in top_ks
        ]
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            label=legend_labels[method],
            zorder=2,
        )
        add_bar_labels(ax, bars, fmt="{:.1f}", offset=1.1, fontsize=6.0)

    ax.set_xticks(x, [f"Top-{top_k}" for top_k in top_ks])
    ax.set_ylim(0, 90)
    ax.set_yticks((0, 20, 40, 60, 80))
    ax.set_ylabel("Structure match rate (%)")
    context = "free" if lattice == "Lattice-free" else "known"
    ax.set_title(
        f"Fixed >20 subset: {context} (n=500)",
        loc="left",
        pad=7,
        fontsize=8.2,
    )
    quiet_axis(ax, grid_axis="y")
    panel_label(ax, label, x=-0.16, y=1.05)
    if show_legend:
        ax.legend(
            loc="upper left",
            ncol=1,
            handlelength=1.3,
            handletextpad=0.4,
            borderaxespad=0.3,
            fontsize=6.0,
        )


def _plot_gain_panel(
    ax,
    rows: list[dict[str, str]],
    *,
    label: str,
) -> None:
    top_ks = (1, 5, 10, 20, 50)
    x = np.arange(len(top_ks), dtype=float)
    specifications = (
        (
            "Lattice-free",
            "Specialist - base multimodal",
            PALETTE["blue"],
            "-",
            "o",
            "free: vs base multi.",
        ),
        (
            "Lattice-free",
            "Specialist - base formula reference",
            PALETTE["blue"],
            "--",
            "D",
            "free: vs base formula ref.",
        ),
        (
            "Known lattice",
            "Specialist - base multimodal",
            PALETTE["violet"],
            "-",
            "o",
            "known: vs base multi.",
        ),
        (
            "Known lattice",
            "Specialist - base formula reference",
            PALETTE["violet"],
            "--",
            "D",
            "known: vs base formula ref.",
        ),
    )
    for lattice, comparison, color, linestyle, marker, legend_label in specifications:
        gains = [
            _value(
                rows,
                value_column="gain_pp",
                lattice=lattice,
                comparison=comparison,
                top_k=top_k,
            )
            for top_k in top_ks
        ]
        ax.plot(
            x,
            gains,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.4,
            markerfacecolor="white" if linestyle == "--" else color,
            markeredgewidth=0.8,
            linewidth=1.25,
            label=legend_label,
            zorder=3,
        )

    ax.axhline(0, color=PALETTE["neutral_3"], lw=0.75)
    ax.set_xticks(x, [str(top_k) for top_k in top_ks])
    ax.set_xlabel("Top-k")
    ax.set_ylim(0, 16.7)
    ax.set_yticks((0, 4, 8, 12, 16))
    ax.set_ylabel("Specialist gain (percentage points)")
    ax.set_title(
        "Same-subset specialist gain",
        loc="left",
        pad=7,
        fontsize=8.4,
    )
    quiet_axis(ax, grid_axis="y")
    panel_label(ax, label, x=-0.16, y=1.05)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=1,
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.2,
        fontsize=5.8,
    )


def main() -> None:
    apply_style()
    full_rows = _read_rows(FULL_DATA)
    capacity_rows = _read_rows(CAPACITY_DATA)
    subset_rows = _read_rows(SUBSET_DATA)
    gain_rows = _read_rows(GAIN_DATA)

    fig, axes = plt.subplots(2, 3, figsize=(7.35, 6.15))
    _plot_full_test_panel(axes[0, 0], full_rows, top_k=1, label="a")
    _plot_full_test_panel(axes[0, 1], full_rows, top_k=20, label="b")
    _plot_capacity_panel(axes[0, 2], capacity_rows, label="c")
    _plot_subset_bars(
        axes[1, 0],
        subset_rows,
        lattice="Lattice-free",
        label="d",
        show_legend=True,
    )
    _plot_subset_bars(
        axes[1, 1],
        subset_rows,
        lattice="Known lattice",
        label="e",
        show_legend=False,
    )
    _plot_gain_panel(axes[1, 2], gain_rows, label="f")

    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.94,
        bottom=0.105,
        wspace=0.37,
        hspace=0.53,
    )
    source_note(
        fig,
        (
            "a,b: full test size bins (N=33,944; n=30,159/3,785). "
            "d–f: one fixed >20 subset (n=500); cohorts are not pooled. "
            "c: Tier-A train N=3,456,202."
        ),
        y=0.018,
    )
    save_figure(fig, OUTPUT_DIR, "figure_5_large_cell", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 5 to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
