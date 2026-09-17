# Crystal Language Route (parallel to FlowMM)

Autoregressive language-model route for experimental crystal structure solving.
This package is independent from the FlowMM/DiffCSP track. It reuses read-only
multimodal data and writes its own artifacts. See
`../docs/canonical_crystal_language_route.md` for the format spec.

## Environment

Use the `LLMD` conda env (PyTorch 2.6 + CUDA):

```bash
conda activate LLMD
```

Data export only needs the standard library, so it can run in any Python.

## 1. Export sequence data (MP+WBM)

Reads `data_prep/outputs/crystal_native_mp_wbm/{train,valid,test}.jsonl`
(produced for the crystal-native route) and writes sequence JSONL to a separate
directory. Source files are never modified.

```bash
python crystal_language/data/export_sequences.py \
  --in-dir data_prep/outputs/crystal_native_mp_wbm \
  --out-dir data_prep/outputs/crystal_language_mp_wbm
```

Each output record stores the canonical crystal `target` string plus per-modality
condition blocks (`pxrd`, `neutron`, `element`, `xrf`, `xps`, `formula`,
`lattice`). Numbers are rendered as character tokens to keep the vocabulary small.

## 2. Train

```bash
python crystal_language/train.py \
  --data-dir data_prep/outputs/crystal_language_mp_wbm \
  --out-dir crystal_language/runs/mp_wbm_v1 \
  --condition-set mix --use-lattice mix \
  --block-size 1024 --batch-size 16
```

- `--condition-set`: one of the ablation sets or `mix` (samples per example for
  condition dropout). Sets: `pxrd`, `pxrd_element`, `pxrd_xrf`, `pxrd_neutron`,
  `pxrd_neutron_xrf`, `pxrd_neutron_xrf_xps`, `pxrd_formula`.
- `--use-lattice`: `always`, `never`, or `mix` (covers the lattice+conditions vs
  conditions-without-lattice comparison).
- `--smoke`: tiny model + few steps for a quick end-to-end check.

The tokenizer is built from the training split on first run and saved to the run
directory as `tokenizer.json`.

## 3. Generate / parse

```bash
python crystal_language/generate.py \
  --run-dir crystal_language/runs/mp_wbm_v1 \
  --data-dir data_prep/outputs/crystal_language_mp_wbm \
  --split valid --index 0 \
  --condition-set pxrd_formula --use-lattice \
  --num-samples 4
```

Generated token sequences are parsed by `crystal_language/decode.py`. Use
`decode.to_structure(parsed)` to build a `pymatgen.Structure` for
`StructureMatcher` and PXRD re-simulation rewards.

## Ablation ladder

Train one model with `--condition-set mix --use-lattice mix`, then evaluate per
condition set by generating with a fixed `--condition-set` and lattice toggle.
This reproduces the project ladder:

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

## Notes

- Large unit cells produce long sequences (an 80-atom cell is ~1900 tokens).
  Increase `--block-size` or filter by `n_sites` for full training.
- This route must stay parallel to FlowMM: do not modify FlowMM code, configs,
  checkpoints, or its data caches.
