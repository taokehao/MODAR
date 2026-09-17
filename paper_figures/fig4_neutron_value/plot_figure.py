#!/usr/bin/env python3
"""Render manuscript Figure 4: the value of neutron diffraction."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
DATA_DIR = FIGURE_DIR / "data"
OUTPUT_DIR = FIGURE_DIR / "outputs"
sys.path.insert(0, str(REPO_ROOT))

from paper_figures.common.style import (  # noqa: E402
    PALETTE,
    apply_style,
    draw_peak_pattern,
    panel_label,
    quiet_axis,
    save_figure,
    source_note,
)


SYSTEM_ORDER = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]
SYSTEM_SHORT = {
    "triclinic": "Tri",
    "monoclinic": "Mono",
    "orthorhombic": "Ortho",
    "tetragonal": "Tetra",
    "trigonal": "Trig",
    "hexagonal": "Hex",
    "cubic": "Cubic",
}
LATTICE_ORDER = ["Lattice-free", "Known lattice"]
LATTICE_COLORS = {
    "Lattice-free": PALETTE["teal"],
    "Known lattice": PALETTE["blue"],
}
LATTICE_MARKERS = {"Lattice-free": "o", "Known lattice": "s"}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(REPO_ROOT)}; run prepare_data.py first"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _minus(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _set_panel_title(ax, title: str) -> None:
    ax.set_title(
        title,
        loc="left",
        pad=8,
        fontsize=8.4,
        fontweight="bold",
        color=PALETTE["black"],
    )


def _draw_schematic(ax) -> None:
    ax.set_axis_off()
    _set_panel_title(ax, "Complementary scattering")
    panel_label(ax, "a", x=-0.08, y=1.04)

    ax.add_patch(
        Rectangle(
            (0.01, 0.875),
            0.50,
            0.075,
            transform=ax.transAxes,
            facecolor=PALETTE["neutral_1"],
            edgecolor="none",
        )
    )
    ax.text(
        0.025,
        0.912,
        "SCHEMATIC · NOT DATA",
        transform=ax.transAxes,
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )

    pxrd_ax = ax.inset_axes([0.10, 0.62, 0.86, 0.14])
    pxrd_ax.set_facecolor(PALETTE["blue_pale"])
    draw_peak_pattern(
        pxrd_ax,
        color=PALETTE["blue_2"],
        peaks=[
            (15, 0.28),
            (23, 0.93),
            (32, 0.46),
            (46, 0.70),
            (57, 0.32),
            (68, 0.42),
        ],
        linewidth=1.3,
    )
    ax.add_patch(
        Rectangle(
            (0.035, 0.62),
            0.025,
            0.14,
            transform=ax.transAxes,
            facecolor=PALETTE["blue_2"],
            edgecolor="none",
        )
    )
    ax.text(
        0.035,
        0.80,
        "PXRD",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=PALETTE["blue"],
    )
    ax.text(
        0.10,
        0.57,
        "electron density · Z-weighted contrast",
        transform=ax.transAxes,
        fontsize=5.9,
        color=PALETTE["neutral_4"],
    )

    nd_ax = ax.inset_axes([0.10, 0.34, 0.86, 0.14])
    nd_ax.set_facecolor(PALETTE["teal_pale"])
    draw_peak_pattern(
        nd_ax,
        color=PALETTE["teal"],
        peaks=[
            (15, 0.78),
            (23, 0.34),
            (32, 0.90),
            (46, 0.38),
            (57, 0.67),
            (68, 0.25),
        ],
        linewidth=1.3,
    )
    ax.add_patch(
        Rectangle(
            (0.035, 0.34),
            0.025,
            0.14,
            transform=ax.transAxes,
            facecolor=PALETTE["teal"],
            edgecolor="none",
        )
    )
    ax.text(
        0.035,
        0.52,
        "Neutron diffraction (ND)",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=PALETTE["teal"],
    )
    ax.text(
        0.10,
        0.29,
        "nuclear contrast · isotope sensitive",
        transform=ax.transAxes,
        fontsize=5.9,
        color=PALETTE["neutral_4"],
    )

    ax.annotate(
        "",
        xy=(0.50, 0.14),
        xytext=(0.50, 0.26),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["neutral_3"], "lw": 0.9},
    )
    ax.text(
        0.50,
        0.07,
        "Joint constraints from complementary\npeak intensities",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=PALETTE["black"],
    )


def _draw_forest(ax, paired: list[dict[str, str]]) -> None:
    _set_panel_title(ax, "Paired neutron gains (95% CI)")
    panel_label(ax, "b", x=-0.16, y=1.04)
    ordered = sorted(
        paired,
        key=lambda row: (int(row["comparison_order"]), int(row["lattice_order"])),
    )
    y_positions = np.arange(len(ordered))[::-1]

    ax.axvline(0, color=PALETTE["black"], lw=1.15, zorder=1)
    ax.axhline(1.5, color=PALETTE["neutral_2"], lw=0.7, zorder=0)
    for y, row in zip(y_positions, ordered):
        estimate = _float(row, "estimate_pp")
        low = _float(row, "ci_low_pp")
        high = _float(row, "ci_high_pp")
        lattice = row["lattice_mode"]
        ax.errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt=LATTICE_MARKERS[lattice],
            ms=5.0,
            mfc=LATTICE_COLORS[lattice],
            mec="white",
            mew=0.6,
            ecolor=LATTICE_COLORS[lattice],
            elinewidth=1.4,
            capsize=2.2,
            capthick=1.0,
            zorder=3,
        )
        ax.text(
            min(high + 0.28, 10.25),
            y,
            _minus(estimate),
            ha="left",
            va="center",
            fontsize=5.9,
            color=PALETTE["neutral_4"],
        )

    labels = [
        f"{row['comparison']} · "
        f"{'Free' if row['lattice_mode'] == 'Lattice-free' else 'Known'}"
        for row in ordered
    ]
    ax.set_yticks(y_positions, labels)
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlim(-1.4, 12.35)
    ax.set_xticks([-1, 0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("ΔTop-1 (percentage points)")
    ax.tick_params(axis="y", length=0, labelsize=6.1, pad=2)
    quiet_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)


def _system_matrix(
    systems: list[dict[str, str]], value_key: str
) -> np.ndarray:
    lookup = {
        (row["lattice_mode"], row["crystal_system"]): _float(row, value_key)
        for row in systems
    }
    return np.array(
        [
            [lookup[(lattice, system)] for system in SYSTEM_ORDER]
            for lattice in LATTICE_ORDER
        ]
    )


def _draw_heatmap(
    ax,
    systems: list[dict[str, str]],
    *,
    value_key: str,
    title: str,
    label: str,
    cmap,
    norm,
) -> None:
    _set_panel_title(ax, title)
    panel_label(ax, label, x=-0.17, y=1.04)
    matrix = _system_matrix(systems, value_key)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(
        range(len(SYSTEM_ORDER)),
        [SYSTEM_SHORT[system] for system in SYSTEM_ORDER],
        rotation=43,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(LATTICE_ORDER)), ["Free", "Known"])
    ax.set_xticks(np.arange(-0.5, len(SYSTEM_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LATTICE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = 0.62 * max(abs(float(norm.vmin)), abs(float(norm.vmax)))
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = "white" if abs(value) >= threshold else PALETTE["black"]
            ax.text(
                col_index,
                row_index,
                _minus(value, 1),
                ha="center",
                va="center",
                fontsize=5.0,
                color=color,
                fontweight="bold" if abs(value) >= threshold else "normal",
            )
def _draw_baseline_gain(ax, systems: list[dict[str, str]]) -> None:
    _set_panel_title(ax, "Baseline vs neutron gain")
    panel_label(ax, "e", x=-0.17, y=1.04)
    maximum_n = max(int(row["n"]) for row in systems)

    ax.axhline(0, color=PALETTE["black"], lw=1.15, zorder=1)
    for lattice in LATTICE_ORDER:
        rows = sorted(
            (row for row in systems if row["lattice_mode"] == lattice),
            key=lambda row: int(row["system_order"]),
        )
        color = LATTICE_COLORS[lattice]
        marker = LATTICE_MARKERS[lattice]
        for row in rows:
            baseline = _float(row, "pxrd_top1_pct")
            gain = _float(row, "neutron_gain_pp")
            n = int(row["n"])
            marker_size = 18.0 + 112.0 * n / maximum_n
            ax.vlines(
                baseline,
                0,
                gain,
                color=color,
                linewidth=0.8,
                alpha=0.42,
                zorder=1,
            )
            ax.scatter(
                baseline,
                gain,
                s=marker_size,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.88,
                zorder=3,
            )
            if row["crystal_system"] in {"hexagonal", "cubic"}:
                mode_short = "F" if lattice == "Lattice-free" else "K"
                offset = (3, 3) if row["crystal_system"] == "hexagonal" else (3, -8)
                ax.annotate(
                    f"{SYSTEM_SHORT[row['crystal_system']]}-{mode_short}",
                    (baseline, gain),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=5.5,
                    color=PALETTE["neutral_4"],
                )

    handles = [
        Line2D(
            [0],
            [0],
            marker=LATTICE_MARKERS[lattice],
            linestyle="",
            markerfacecolor=LATTICE_COLORS[lattice],
            markeredgecolor="white",
            markersize=6,
            label="Free" if lattice == "Lattice-free" else "Known",
        )
        for lattice in LATTICE_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, handletextpad=0.3)
    ax.text(
        0.98,
        0.03,
        "Bubble size encodes n",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["neutral_4"],
    )
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.9, 17.2)
    ax.set_xlabel("PXRD baseline Top-1 (%)")
    ax.set_ylabel("ND vs PXRD gain (pp)")
    quiet_axis(ax, grid_axis="both")


def _summary_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    heading: str,
    value: str,
    status: str,
    accent: str,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=PALETTE["neutral_2"],
            linewidth=0.65,
        )
    )
    ax.text(
        x + 0.035,
        y + height - 0.04,
        heading,
        transform=ax.transAxes,
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        x + 0.035,
        y + height - 0.13,
        value,
        transform=ax.transAxes,
        va="top",
        fontsize=5.35,
        linespacing=1.2,
        color=PALETTE["black"],
    )
    ax.text(
        x + 0.035,
        y + 0.025,
        status,
        transform=ax.transAxes,
        va="bottom",
        fontsize=5.5,
        fontweight="bold",
        color=accent,
    )


def _draw_summary(
    ax, systems: list[dict[str, str]], paired: list[dict[str, str]]
) -> None:
    ax.set_axis_off()
    _set_panel_title(ax, "Beneficiary summary")
    panel_label(ax, "f", x=-0.08, y=1.04)

    ax.text(
        0.02,
        0.88,
        "Largest crystal-system gains · ND vs PXRD",
        transform=ax.transAxes,
        fontsize=6.7,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )
    y = 0.79
    for lattice in LATTICE_ORDER:
        top_rows = sorted(
            (row for row in systems if row["lattice_mode"] == lattice),
            key=lambda row: _float(row, "neutron_gain_pp"),
            reverse=True,
        )[:3]
        values = "  ·  ".join(
            f"{SYSTEM_SHORT[row['crystal_system']]} {_minus(_float(row, 'neutron_gain_pp'))}"
            for row in top_rows
        )
        mode = "Free" if lattice == "Lattice-free" else "Known"
        ax.text(
            0.02,
            y,
            mode,
            transform=ax.transAxes,
            fontsize=6.6,
            fontweight="bold",
            color=LATTICE_COLORS[lattice],
        )
        ax.text(
            0.27,
            y,
            f"{values} pp",
            transform=ax.transAxes,
            fontsize=5.5,
            color=PALETTE["black"],
        )
        y -= 0.10

    ax.plot(
        [0.02, 0.98],
        [0.57, 0.57],
        transform=ax.transAxes,
        color=PALETTE["neutral_2"],
        lw=0.7,
    )
    ax.text(
        0.02,
        0.51,
        "Aggregate ND given XRF · paired N=10,000",
        transform=ax.transAxes,
        fontsize=6.7,
        fontweight="bold",
        color=PALETTE["neutral_4"],
    )

    lookup = {
        row["lattice_mode"]: row
        for row in paired
        if row["comparison"] == "ND given XRF"
    }
    free = lookup["Lattice-free"]
    known = lookup["Known lattice"]
    _summary_box(
        ax,
        (0.02, 0.12),
        0.45,
        0.34,
        facecolor=PALETTE["teal_pale"],
        heading="Lattice-free",
        value=(
            f"{_minus(_float(free, 'estimate_pp'))} pp\n"
            f"95% CI [{_minus(_float(free, 'ci_low_pp'))}, "
            f"{_minus(_float(free, 'ci_high_pp'))}]"
        ),
        status="CI crosses zero",
        accent=PALETTE["teal"],
    )
    _summary_box(
        ax,
        (0.52, 0.12),
        0.46,
        0.34,
        facecolor=PALETTE["blue_pale"],
        heading="Known lattice",
        value=(
            f"{_minus(_float(known, 'estimate_pp'))} pp\n"
            f"95% CI [{_minus(_float(known, 'ci_low_pp'))}, "
            f"{_minus(_float(known, 'ci_high_pp'))}]"
        ),
        status="Significant positive",
        accent=PALETTE["blue"],
    )
    ax.text(
        0.02,
        0.08,
        "Crystal-system gains are descriptive; subgroup CIs are not available in the source.",
        transform=ax.transAxes,
        fontsize=5.6,
        color=PALETTE["neutral_3"],
    )


def main() -> None:
    apply_style()
    paired = _load_csv(DATA_DIR / "paired_neutron_effects.csv")
    systems = _load_csv(DATA_DIR / "crystal_system_neutron_effects.csv")

    gain_cmap = LinearSegmentedColormap.from_list(
        "neutron_gain",
        [
            PALETTE["neutral_0"],
            PALETTE["teal_pale"],
            PALETTE["teal"],
            PALETTE["blue"],
        ],
    )
    residual_cmap = LinearSegmentedColormap.from_list(
        "neutron_given_xrf",
        [PALETTE["red"], PALETTE["neutral_0"], PALETTE["blue"]],
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.35, 6.25),
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.95,
        bottom=0.105,
        wspace=0.68,
        hspace=0.63,
    )

    _draw_schematic(axes[0, 0])
    _draw_forest(axes[0, 1], paired)
    _draw_heatmap(
        axes[0, 2],
        systems,
        value_key="neutron_gain_pp",
        title="ND gain by system (pp)",
        label="c",
        cmap=gain_cmap,
        norm=Normalize(vmin=0, vmax=16),
    )
    _draw_heatmap(
        axes[1, 0],
        systems,
        value_key="neutron_given_xrf_gain_pp",
        title="ND gain given XRF (pp)",
        label="d",
        cmap=residual_cmap,
        norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5),
    )
    _draw_baseline_gain(axes[1, 1], systems)
    _draw_summary(axes[1, 2], systems, paired)

    source_note(
        fig,
        "Source: ladder_stats.json, aligned N=10,000 evaluation subset. Error bars are 95% paired bootstrap CIs "
        "(2,000 resamples), not full-test CIs.\nPanel a is schematic (not data); all gains are percentage points.",
        y=0.018,
    )
    save_figure(fig, OUTPUT_DIR, "figure_4_neutron_value", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
