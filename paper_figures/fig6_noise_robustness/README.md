# Figure 6 — robustness to synthetic experimental artefacts

This package reproduces the manuscript 3×2 Figure 6 and its curated source-data
tables. “Noisy” means online synthetic perturbations of simulated inputs
(shift, broadening, intensity/texture changes, background, and counting noise);
it does **not** mean measurements from real experiments. Every reported result
uses `N=500` samples per configuration.

## Panel map

- **a** — qualitative schematic of the synthetic artefact pipeline. The traces
  are illustrative and are not experimental observations or quantitative data.
- **b** — clean versus noisy Top-1 for the three high-information conditions,
  shown for lattice-free and known-lattice inference.
- **c** — the corresponding Top-20 values, highlighting that noisy recall stays
  at or above 92.4% for the displayed conditions.
- **d** — baseline `σ=0.25` versus deploy `σ=0.40` with `±0.20°` shift search
  on the same frozen base candidate pool. This policy is shown only for noisy,
  lattice-free inference.
- **e** — baseline versus Rietveld-lite refined Top-1 within the base refinement
  pool; XPS is visually de-emphasized. The big-cell result is a separate diamond
  row from its own candidate pool.
- **f** — clean/noisy big-cell comparison plus the three main robustness
  takeaways.

## Source results

`prepare_data.py` reads only:

- `crystal_language/eval_results/noisy_robustness_summary.json`
- `crystal_language/eval_results/noisy_rerank_deploy_base_ladder.json`
- `crystal_language/eval_results/refine_rerank_ablation_base_free.json`
- `crystal_language/eval_results/refine_rerank_ablation_bigcell_free.json`

It writes:

- `data/clean_noisy_high_information.csv`
- `data/deploy_rerank_free.csv`
- `data/refinement_free.csv`
- `data/bigcell_clean_noisy.csv`

Each deploy or refinement row carries a `pool_id`. Deltas are computed only
between the baseline and intervention evaluated on that same frozen candidate
pool. Baselines from different JSON files are intentionally not added together.

The clean/noisy ladder comparison is the published system-level contrast:
clean-trained model plus clean input versus noise-finetuned model plus noisy
input. It should not be interpreted as a pure input-noise ablation.

## Rebuild

From the repository root, with NumPy and Matplotlib available:

```bash
python paper_figures/fig6_noise_robustness/prepare_data.py
python paper_figures/fig6_noise_robustness/plot_figure.py
```

The local rendering used the existing environment:

```bash
conda run -n pxrd_crystal_lm python \
  paper_figures/fig6_noise_robustness/plot_figure.py
```

The plotting code imports `paper_figures/common/style.py` and exports:

- `outputs/figure_6_noise_robustness.png` (300 dpi)
- `outputs/figure_6_noise_robustness.pdf`
- `outputs/figure_6_noise_robustness.svg`
