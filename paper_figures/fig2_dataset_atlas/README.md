# Figure 2 — Dataset atlas

This directory contains the complete, reproducible data preparation and
rendering workflow for manuscript Figure 2. The figure canvas is 7.2 × 7.9
inches and uses the shared style in `paper_figures/common/style.py`.

## Reproduce

Run from the repository root:

```bash
python paper_figures/fig2_dataset_atlas/prepare_data.py
python paper_figures/fig2_dataset_atlas/plot_figure.py
```

`prepare_data.py` uses only the Python standard library. `plot_figure.py`
requires NumPy and Matplotlib. On this workstation it was verified with the
`pxrd_crystal_lm` Conda environment:

```bash
conda run -n pxrd_crystal_lm python \
  paper_figures/fig2_dataset_atlas/plot_figure.py
```

The commands generate:

- `outputs/figure_2_dataset_atlas.png` (300 dpi)
- `outputs/figure_2_dataset_atlas.pdf`
- `outputs/figure_2_dataset_atlas.svg`

## Primary sources

- `data_prep/outputs/metadata_index_unified_tight/metadata_index_summary.json`
  supplies the governance funnel and final Tier-A kept source composition.
- `data_prep/outputs/crystal_language_unified_tierA_v2/export_summary.json`
  supplies the frozen train/valid/test sizes.
- `data_prep/outputs/crystal_language_unified_tierA_v2/test.jsonl` supplies all
  test-set distributions. The script streams all 33,944 records; it does not
  sample.

`data/provenance.json` records SHA-256 hashes for all three inputs.

## Parsing and definitions

- Elements are parsed from each target's `<FORMULA> ... <N_SITES>` segment.
  Prevalence is the percentage of test records containing an element, with an
  element counted at most once per record.
- Multi-digit formula amounts and `<N_SITES>` values are digit-tokenized in the
  target (for example, `12` is represented as `1 2`); the parser joins those
  tokens before validation.
- `n_sites` is read from the record field and cross-checked against target
  `<N_SITES>` for every record.
- `n_elements` is the number of distinct formula elements.
- Crystal systems are mapped from the single target `<SG###>` token using the
  standard ranges: triclinic 1–2, monoclinic 3–15, orthorhombic 16–74,
  tetragonal 75–142, trigonal 143–167, hexagonal 168–194, and cubic 195–230.

## Prepared data

- `data/data_governance_funnel.csv`
- `data/source_composition.csv`
- `data/split_sizes.csv`
- `data/element_prevalence.csv` (all 118 elements, including zero-prevalence
  entries)
- `data/n_sites_distribution.csv` (all integer values from 1 through 128)
- `data/n_elements_distribution.csv`
- `data/crystal_system_distribution.csv`
- `data/dataset_summary.json`
- `data/provenance.json`

## Verified values

- Funnel: 10,207,687 unified records → 4,127,590 Tier-A records → 3,524,185
  records after tight deduplication.
- Frozen split: train 3,456,202; valid 34,039; test 33,944.
- Final kept source mix: Alexandria 62.602%, AFLOW 15.307%, OQMD 10.763%,
  WBM 7.099%, and MP 4.230%.
- Test scan: 33,944 of 33,944 records processed; 85 elements present.
- Test structure size: median 10 sites, mean 12.781, 95th percentile 36, range
  1–128.
- Test composition size: median 3 distinct elements, mean 3.313, range 1–8.
- Test crystal systems: triclinic 1,502; monoclinic 8,104; orthorhombic 7,676;
  tetragonal 7,187; trigonal 5,325; hexagonal 2,096; cubic 2,054.

Panel f is explicitly schematic. Its XRF branch represents structured element
mass fractions, not a raw energy spectrum.
