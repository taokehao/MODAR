# Figure 5 — Large-cell specialization

This package reproduces the six-panel manuscript figure showing why cell-size
specialization is needed and how targeted routing improves large-cell
reconstruction.

## Panel map

- **a** — Top-1 reranked structure-match rate on the full test split
  (`N=33,944`), separated into `≤20` (`n=30,159`) and `>20`
  (`n=3,785`) cells. Base PXRD+ND+XRF is compared with the base formula
  reference under lattice-free and known-lattice settings.
- **b** — The same full-test comparison at Top-20.
- **c** — Tier-A training composition: `3,093,529` `≤20` cells versus
  `362,673` `>20` cells (`N=3,456,202`), followed by the inference-time
  routing rule (`≤20` to the base model, `>20` to the large-cell specialist).
- **d** — Top-1 and Top-20 on one fixed `>20` test subset (`n=500`) under
  lattice-free generation: base multimodal, specialist multimodal, and base
  formula reference.
- **e** — The same fixed 500-cell subset with known lattice.
- **f** — Specialist gains over the base multimodal model and over the base
  formula reference at Top-1/5/10/20/50, always computed within that same
  500-cell subset.

Panels a–b and panels d–f are different evaluation cohorts. The full size-bin
counts (`30,159` and `3,785`) are never pooled with or treated as paired to the
fixed 500-cell comparison subset.

## Evidence statement

On exactly the same 500 large-cell examples, the specialist with
PXRD+ND+XRF exceeds the base formula reference: Top-1 is `66.4%` versus
`58.4%` lattice-free (`+8.0` percentage points) and `72.6%` versus `61.0%`
with known lattice (`+11.6` percentage points). This is a same-subset,
same-metric model comparison.

## Source provenance

- `crystal_language/eval_results/ladder_tierA_s2_full_neutron/size_bins_20.json`
  supplies the full-test size-bin rates and sample counts for panels a–b.
- `crystal_language/eval_results/bigcell_ab/oracle_same_subset_comparison.json`
  supplies all fixed-subset rates for panels d–f. The preparation script maps
  the formula-constrained source key to the manuscript label “Base formula
  reference.”
- `总体实施方案.md` §8.10.1 supplies the `>20` training count (`362,673`)
  and routing provenance; §8.7.1 cross-checks the total Tier-A train count
  (`3,456,202`). The `≤20` count is their difference.

All plotted match rates are percentages. Panel-f gains are percentage-point
differences computed by `prepare_data.py`, not separately entered values.

## Rebuild

Run from the repository root:

```bash
python paper_figures/fig5_large_cell/prepare_data.py
python paper_figures/fig5_large_cell/plot_figure.py
```

The preparation step writes:

- `data/full_test_size_bins.csv`
- `data/train_capacity.csv`
- `data/bigcell_same_subset.csv`
- `data/bigcell_gains.csv`

The rendering step reuses `paper_figures/common/style.py` and writes:

- `outputs/figure_5_large_cell.png`
- `outputs/figure_5_large_cell.pdf`
- `outputs/figure_5_large_cell.svg`
