# Figure 4 — Value of neutron diffraction

This package builds the manuscript 2 × 3 Figure 4 from the aligned 10,000-sample
Tier-A evaluation subset.

## Panel map

- **a** — Pure schematic of complementary PXRD and neutron scattering contrast.
  The traces are illustrative, carry no numerical result, and are labelled
  `SCHEMATIC · NOT DATA`.
- **b** — Paired ΔTop-1 forest plot for ND vs PXRD and ND given XRF, under
  lattice-free and known-lattice settings. Error bars are 95% paired bootstrap
  confidence intervals.
- **c** — Crystal-system heatmap of
  `Top-1(PXRD + ND) − Top-1(PXRD)`.
- **d** — Crystal-system heatmap of
  `Top-1(PXRD + ND + XRF) − Top-1(PXRD + XRF)`.
- **e** — PXRD baseline Top-1 versus neutron gain. Bubble size encodes the
  number of structures in each crystal system.
- **f** — Summary of the strongest crystal-system beneficiaries and the
  aggregate ND-given-XRF inference: the lattice-free 95% CI crosses zero,
  whereas the known-lattice effect is significantly positive.

All gains are reported in **percentage points**. Crystal systems use one fixed
order throughout: triclinic, monoclinic, orthorhombic, tetragonal, trigonal,
hexagonal, cubic.

## Source and statistical scope

Source artifact:

```text
crystal_language/eval_results/ladder_tierA_s2_10k/stats/ladder_stats.json
```

`prepare_data.py` extracts only fields contained in this artifact. The four
paired effects use the same aligned `N = 10,000` structures and the source
artifact's 2,000-resample paired bootstrap intervals. These intervals describe
the **10k evaluation subset**; they are not full-test confidence intervals.
Crystal-system cells are descriptive point estimates because the source does
not provide subgroup confidence intervals.

Prepared source-data tables:

```text
data/paired_neutron_effects.csv
data/crystal_system_neutron_effects.csv
```

## Rebuild

From the repository root:

```bash
python paper_figures/fig4_neutron_value/prepare_data.py
python paper_figures/fig4_neutron_value/plot_figure.py
```

The plotting code reuses `paper_figures/common/style.py` and writes:

```text
outputs/figure_4_neutron_value.png
outputs/figure_4_neutron_value.pdf
outputs/figure_4_neutron_value.svg
```
