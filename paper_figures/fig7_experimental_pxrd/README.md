# Figure 7 — Experimental powder diffraction

Three-layer Nature-width figure (7.2 × 8.55 in). Uses
`paper_figures/common/style.py`.

## Story

- **a** — CNRS-126 / RRUFF-269 versus RealPXRD-Solver (formula + known lattice,
  refine-before-match). Tied cells are argued from 95% CIs.
- **b** — Public 337 measured Cu Kα **without formula**. Spectroscopic rungs use
  CHGNet ∪ MACE refine. Formula reference is unrefined fusion (formula MACE
  shards are not required). Counts, not an accuracy percentage. ND/XRF are
  CIF-simulated.
- **c** — One laboratory case (LiCoO2). Clean model misses; noise-adapted
  inference lifts PXRD rank 2 to Top-1 with simulated ND.

## Reproduce

```bash
python paper_figures/fig7_experimental_pxrd/prepare_data.py
python paper_figures/fig7_experimental_pxrd/plot_figure.py
```

Verified with `conda run -n pxrd_crystal_lm python paper_figures/fig7_experimental_pxrd/plot_figure.py`.

Writes `outputs/figure_7_experimental_pxrd.{png,pdf,svg}`.

## Sources

- `crystal_language/eval_results/chgnet/rps_cnrs126_bootstrap_refined.json`
- `crystal_language/eval_results/chgnet/rps_rruff269_bootstrap_refined.json`
- `crystal_language/eval_results/chgnet/ladder_public_si_report_spec_cm.json`
- `data/licoo2_spectra.csv` (measured PXRD vs CIF-simulated ND)
- `data/fig7c_licoo2_known_ranks.csv` (LiCoO2 known-lattice ranks)

Related SI: Fig. S17 (lattice-free 337), S18 (n≤20 split), S19 (paired steps),
S20 (XPS increment). The previous keep-10 Fig. 7 panels are not used in the
main figure; Fig. S12 still holds the other collaborator samples.
