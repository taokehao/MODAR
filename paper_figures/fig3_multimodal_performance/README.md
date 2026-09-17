# Figure 3 — multimodal performance

This package builds the 7.2-inch-wide, six-panel Nature-style performance
figure from one project result artifact. It uses the shared visual settings in
`../common/style.py`.

## Provenance and scope

The only numerical source is:

`../../crystal_language/eval_results/ladder_tierA_s2_full_neutron/ladder_test_table.json`

The source covers the full test set (`N=33,944`) with `N=100` generated
candidates per sample. Every success rate shown is from `topk_rerank`;
`topk_sampling` is not used.

The main figure includes exactly these five conditions:

- `pxrd` → PXRD
- `pxrd_neutron` → PXRD + ND
- `pxrd_xrf` → PXRD + XRF
- `pxrd_neutron_xrf` → PXRD + ND + XRF (Multi)
- `pxrd_formula` → Formula reference

The XPS condition is intentionally excluded. Formula reference means
conditioning on the exact formula and is described as an oracle reference,
not a strict theoretical upper bound. The figure is restricted to the five
internal information conditions listed above.

## Panel map

- **a** — Information ladder: PXRD branches through ND and XRF constraints,
  merges into the multimodal condition, and is compared with Formula reference.
- **b** — Lattice-free reranked Top-1 and Top-20 success for all five
  conditions.
- **c** — Known-lattice reranked Top-1 and Top-20 success for the same
  conditions.
- **d** — Composition recovery rate versus reranked Top-1, with conditions
  directly labelled and lattice settings distinguished by marker shape.
- **e** — Reranked Top-k (`k=1,5,10,20,50`) for Multi and Formula reference,
  faceted into lattice-free and known-lattice views.
- **f** — Multi minus Formula-reference gaps at Top-1 and Top-20. The rounded
  comparisons are 84.0 vs 85.9 and 91.5 vs 93.0 for lattice-free; 90.4 vs
  89.5 and 93.6 vs 93.9 for known lattice. The panel also summarizes the
  Top-1 gain of Multi over PXRD.

## Files

- `prepare_data.py` validates and extracts the five conditions into
  `data/figure_3_source_data.csv`.
- `plot_figure.py` reads only that curated CSV and writes the final exports.
- `outputs/figure_3_multimodal_performance.png` is the 300-dpi raster export.
- `outputs/figure_3_multimodal_performance.pdf` and
  `outputs/figure_3_multimodal_performance.svg` are editable vector exports.

## Rebuild

From the repository root:

```bash
python paper_figures/fig3_multimodal_performance/prepare_data.py
python paper_figures/fig3_multimodal_performance/plot_figure.py
```

Both scripts resolve paths relative to their own location, so the commands also
work from another current directory.
