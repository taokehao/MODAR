# FlowMM Adaptation

## Why CDVAE Appears

CDVAE is not the new main model for this project. It appears because FlowMM's
upstream repository keeps CDVAE as a historical submodule for data utilities,
baseline comparisons, and compatibility with earlier crystal-generation
pipelines.

For our project:

- main model target: FlowMM-style Riemannian flow matching;
- stable baseline: DiffCSP-style periodic E(3)-equivariant diffusion;
- CDVAE: optional upstream dependency for full FlowMM reproduction.

The current data adapter does not depend on CDVAE. Missing
`external_models/flowmm/remote/cdvae/` should not block the first project
adaptation step.

## Current Status

- `external_models/flowmm/remote/cdvae/` has been downloaded successfully.
- FlowMM now has a native JSONL data adapter:
  `diffcsp.pl_data.dataset.CrystalNativeJsonlDataset`.
- FlowMM now also has a direct MP+WBM LMDB adapter:
  `diffcsp.pl_data.dataset.CrystalNativeLmdbDataset`.
- The preferred full-dataset project config is:
  `external_models/flowmm/scripts_model/conf/data/pxrd_mof_jsonl.yaml`.
- The preferred quick MP+WBM config is:
  `external_models/flowmm/scripts_model/conf/data/pxrd_mof_mp_wbm_lmdb.yaml`.

## Data Adaptation Stages

### Stage 0: Direct MP+WBM LMDB

For quick FlowMM development, use the already-built MP+WBM LMDB directly:

```text
data_prep/outputs/official_mp_wbm_multimodal/{train,valid,test}.lmdb
```

The direct adapter caches FlowMM graph preprocessing next to the LMDB as
`train_<condition_set>_<use_lattice_condition>.pt` and equivalent valid/test
files. Run from `external_models/flowmm`:

```bash
conda activate flowmm
export PROJECT_ROOT=/home/tkh/python-project/multimodal_csp/external_models/flowmm
export PXRD_MOF_MP_WBM_DATA=/home/tkh/python-project/multimodal_csp/data_prep/outputs/official_mp_wbm_multimodal

# lattice-free, PXRD-only smoke run
python scripts_model/run.py \
  data=pxrd_mof_mp_wbm_lmdb_lattice_free \
  model=conditional_null_params \
  data.condition_set=pxrd \
  data.max_records=2048 \
  train.pl_trainer.fast_dev_run=true \
  test.run_after_fit=false

# lattice-conditioned smoke run
python scripts_model/run.py \
  data=pxrd_mof_mp_wbm_lmdb_lattice_conditioned \
  model=conditional_null_params \
  data.condition_set=pxrd_neutron_xrf_xps \
  data.max_records=2048 \
  train.pl_trainer.fast_dev_run=true \
  test.run_after_fit=false
```

Supported condition sets:

```text
pxrd
pxrd_element
pxrd_xrf
pxrd_neutron
pxrd_neutron_xrf
pxrd_neutron_xrf_xps
pxrd_formula
```

`model=conditional_null_params` encodes these fields into a 492-dimensional
graph condition vector and injects it into CSPNet node features. In
`lattice-conditioned` mode, the true lattice is appended to the condition
vector, the lattice slice of the flow is fixed, and lattice loss is zeroed. In
`lattice-free` mode, the model still generates the lattice and optimizes the
lattice flow loss.

### Stage 1: Crystal-Native JSONL

The primary project format is:

```text
id, atom_types, frac_coords, lattice_matrix, formula, conditions, properties
```

Build it from the unified LMDBs with:

```bash
python data_prep/official_multisource/scripts/manage_dataset.py export-jsonl
```

Run FlowMM directly on this JSONL format with:

```bash
cd external_models/flowmm
export PROJECT_ROOT=/home/tkh/python-project/multimodal_csp/external_models/flowmm
export PXRD_MOF_CRYSTAL_NATIVE_DATA=/home/tkh/python-project/multimodal_csp/data_prep/outputs/crystal_native_unified
python scripts_model/run.py data=pxrd_mof_jsonl model=null_params
```

### Stage 2: Structure-Compatible FlowMM CSV

FlowMM currently reuses DiffCSP's `CrystDataset`, which reads CSV files with:

```text
material_id, pretty_formula, cif, formation_energy_per_atom
```

The script below converts our crystal-native JSONL into that format:

```bash
python data_prep/official_multisource/scripts/export_flowmm_csv.py \
  --in-dir data_prep/outputs/crystal_native_unified \
  --out-dir data_prep/outputs/flowmm_unified
```

It also writes sidecar condition files:

```text
data_prep/outputs/flowmm_unified/conditions/train.jsonl
data_prep/outputs/flowmm_unified/conditions/valid.jsonl
data_prep/outputs/flowmm_unified/conditions/test.jsonl
```

These sidecars preserve PXRD, neutron, XRF mass fractions, XPS valence,
properties, and lattice metadata keyed by `material_id`.

### Stage 3: Spectrum-Conditioned FlowMM

The current conditional model uses a lightweight first-pass condition encoder:

```text
PXRD summary + optional neutron summary + element set + XRF + XPS-valence
+ exact-formula composition + optional lattice parameters
```

This lets us reproduce the original comparison ladder:

```text
PXRD only
PXRD + element set
PXRD + XRF
PXRD + neutron
PXRD + neutron + XRF
PXRD + neutron + XRF + XPS-valence
PXRD + exact formula
lattice + conditions
conditions without lattice
```

Run the ladder by changing only `data.condition_set` and choosing one of the
two data configs:

```bash
for condition_set in \
  pxrd pxrd_element pxrd_xrf pxrd_neutron \
  pxrd_neutron_xrf pxrd_neutron_xrf_xps pxrd_formula
do
  python scripts_model/run.py \
    data=pxrd_mof_mp_wbm_lmdb_lattice_free \
    model=conditional_null_params \
    data.condition_set=${condition_set}
done
```

## FlowMM Configs

The project adds these FlowMM data configs:

```text
external_models/flowmm/scripts_model/conf/data/pxrd_mof_mp_wbm_lmdb.yaml
external_models/flowmm/scripts_model/conf/data/pxrd_mof_mp_wbm_lmdb_lattice_free.yaml
external_models/flowmm/scripts_model/conf/data/pxrd_mof_mp_wbm_lmdb_lattice_conditioned.yaml
external_models/flowmm/scripts_model/conf/data/pxrd_mof_jsonl.yaml
external_models/flowmm/scripts_model/conf/data/pxrd_mof.yaml
external_models/flowmm/scripts_model/conf/model/conditional_null_params.yaml
```

Prefer `pxrd_mof_mp_wbm_lmdb_lattice_free` and
`pxrd_mof_mp_wbm_lmdb_lattice_conditioned` for quick MP+WBM development. Prefer
`pxrd_mof_jsonl` once moving to the full unified dataset. Use `pxrd_mof` only
when testing strict DiffCSP-compatible CSV behavior.

After exporting CSV files, the CSV fallback can be run from
`external_models/flowmm` with:

```bash
export PROJECT_ROOT=/home/tkh/python-project/multimodal_csp/external_models/flowmm
export PXRD_MOF_FLOWMM_DATA=/home/tkh/python-project/multimodal_csp/data_prep/outputs/flowmm_unified
python scripts_model/run.py data=pxrd_mof model=null_params
```

`model=conditional_null_params` is the first multimodal conditional setting:
atom types are treated as known/conditioned, while fractional coordinates and,
for lattice-free runs, lattice parameters are generated on crystal-native
manifolds. MP+WBM currently reuses FlowMM's existing MP20 lattice-stat key for
the lattice manifold; compute project-specific lattice stats before full-scale
final training.
