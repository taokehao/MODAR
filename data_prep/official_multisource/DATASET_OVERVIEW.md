# Unified Multimodal Crystal Dataset

This is the canonical data note for the project. The data has already been
downloaded, enriched, built into LMDB, and merged. Do not rerun download or
build scripts unless intentionally regenerating the dataset.

## Final Training Dataset

Primary dataset:

```text
data_prep/outputs/official_unified_multimodal/
  train.lmdb
  valid.lmdb
  test.lmdb
  merge_report.json
  normalization_stats.json
```

Final split sizes from `merge_report.json`:

```text
train  10,003,856
valid     101,978
test      101,853
```

Final train source counts from `normalization_stats.json`:

```text
alexandria  5,809,184
oqmd        1,363,213
aflow       2,429,759
mp            151,718
wbm           249,982
```

MP and WBM together contribute `401,700` train structures after source-level
deduplication. The unified merge performs additional cross-source deduplication,
so source-level build counts and final unified counts are not expected to match
exactly.

## Record Schema

Each LMDB record stores the same model-facing fields, regardless of source:

```text
structure:
  atom_type
  atom_pos
  lattice_matrix

spectra and chemical conditions:
  pxrd_x
  pxrd_y
  neutron_x
  neutron_y
  valence_assignments
  valence_source
  valence_assigned

composition-derived condition:
  XRF mass fraction is derived from atom_type by loaders/exporters.

identity and metadata:
  source
  source_id
  formula
  pretty_formula
  n_sites
  space_group
  structure_hash

properties:
  formation_energy_per_atom
  e_above_hull
  band_gap
  bvse_li_e1d
  bvse_na_e1d
```

This corresponds to the intended condition set:

```text
structure
PXRD
neutron diffraction
XRF mass%
XPS-valence
element set
formula ratio
properties
```

## Source Data Status

Raw and enhanced source files are under:

```text
data_prep/raw/official_mp_wbm/
data_prep/raw/official_extra_sources/
```

The relevant completed source assets are:

```text
MP + WBM:
  data_prep/outputs/official_mp_wbm_multimodal/

Alexandria:
  data_prep/raw/official_extra_sources/alexandria_pbe.jsonl
  data_prep/raw/official_extra_sources/alexandria_pbe.multimodal.jsonl

OQMD:
  data_prep/raw/official_extra_sources/oqmd_formationenergy_merged.jsonl
  data_prep/raw/official_extra_sources/oqmd_formationenergy_merged.multimodal.jsonl

AFLOW:
  data_prep/raw/official_extra_sources/aflow_lib*_raw_shard_*_of_*.jsonl
  data_prep/raw/official_extra_sources/aflow_lib*_raw_shard_*_of_*.multimodal.jsonl
```

Intermediate per-source LMDB build reports are preserved under:

```text
data_prep/outputs/official_extra_sources_parts/
data_prep/outputs/official_aflow_parts/
data_prep/outputs/official_mp_wbm_multimodal/
```

These are provenance artifacts, not the default model input.

## Build History

The historical pipeline was:

```text
MP + WBM:
  01_download_mp_wbm.py
  02_build_lmdb_mp_wbm.py
  07_enrich_lmdb_multimodal.py

Alexandria + OQMD:
  01_download_extra_sources.py
  06_enrich_raw_multimodal.py
  02_build_lmdb_extra_sources.py  # originally parallelized over 16 parts

AFLOW:
  01_download_extra_sources.py
  06_enrich_raw_multimodal.py
  02_build_lmdb_aflow.py          # originally parallelized over 8 parts

Unified merge:
  05_merge_unified_lmdb.py
  03_compute_prop_stats.py
  04_inspect_lmdb.py
```

The parallel part layout was only for download/build throughput. For future
model work, use the merged LMDB or exported crystal-native JSONL.

## Recommended Commands

Check the final dataset:

```bash
python data_prep/official_multisource/scripts/manage_dataset.py status
```

Inspect a few LMDB records:

```bash
python data_prep/official_multisource/scripts/manage_dataset.py inspect \
  --split train --n 3
```

Export the unified LMDB to the crystal-native JSONL bridge format:

```bash
python data_prep/official_multisource/scripts/manage_dataset.py export-jsonl
```

Default output:

```text
data_prep/outputs/crystal_native_unified/
  train.jsonl
  valid.jsonl
  test.jsonl
  bakeoff_protocol.json
  export_summary.json
```

For smoke tests:

```bash
python data_prep/official_multisource/scripts/manage_dataset.py export-jsonl \
  --out-dir data_prep/outputs/crystal_native_unified_smoke \
  --max-records-per-split 1000
```

FlowMM should use the JSONL adapter directly:

```bash
cd external_models/flowmm
export PROJECT_ROOT=/home/tkh/python-project/multimodal_csp/external_models/flowmm
export PXRD_MOF_CRYSTAL_NATIVE_DATA=/home/tkh/python-project/multimodal_csp/data_prep/outputs/crystal_native_unified
python scripts_model/run.py data=pxrd_mof_jsonl model=null_params
```

The MP+WBM-only dataset remains useful for baseline comparison, but it is not
the default data source for the new crystal-native mainline.
