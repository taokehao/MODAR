# MODAR

**M**ultimodal **O**ptional-prefix **D**iffraction **A**utoregressive **R**etrieval

MODAR generates crystal structures from independently available powder X-ray
diffraction (PXRD), neutron diffraction (ND), and elemental-composition inputs.
A shared crystal language supports optional chemical-formula and lattice
constraints, autoregressive candidate generation, and physical ranking.

- **Code:** [taokehao/MODAR](https://github.com/taokehao/MODAR)
- **Trained models:** [Coooolin/MODAR](https://huggingface.co/Coooolin/MODAR)
- **Data and numerical source data:** [Coooolin/MODAR-data](https://huggingface.co/datasets/Coooolin/MODAR-data)

## Installation

Use Python 3.11. Run the commands below from the repository root.
The model code has been checked with PyTorch 2.6.0 and pymatgen 2025.3.10.
A CUDA GPU is recommended for candidate generation and full evaluation;
CPU execution is supported for small checks.

```bash
git clone https://github.com/taokehao/MODAR.git
cd MODAR
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Download a model and evaluation data

These pinned revisions identify the released artifacts. No Hugging Face login
is required for downloading them.

```bash
export MODAR_MODEL_REV=c9520488588062d4f6c6c86cc85fa86516ac1557
export MODAR_DATA_REV=d2cec69e38f5b99cead028de585a2615df311def

hf download Coooolin/MODAR --revision "$MODAR_MODEL_REV" \
  --include 'models/modar-cnn/*' --include MANIFEST.json \
  --local-dir artifacts/model

hf download Coooolin/MODAR-data --repo-type dataset \
  --revision "$MODAR_DATA_REV" \
  --include SPLITS.json --include MANIFEST.json --include SOURCES.md \
  --include 'data/tier-a/test-*.jsonl.gz' --include 'benchmarks/*' \
  --include 'source_data/*' --include 'audits/*' \
  --local-dir artifacts/data

python scripts/materialize_dataset.py \
  --dataset-dir artifacts/data --config tier-a --split test \
  --output-dir artifacts/inputs/tier-a
```

`materialize_dataset.py` concatenates decompressed shards in the order recorded
in `SPLITS.json`, verifies the original stream's SHA-256 and record count, and
writes `test.jsonl`. It can be rerun safely on an already verified output.
The prepared experimental benchmarks are uncompressed and need no conversion.

Each model directory contains `checkpoint_last.pt`, `config.json`, and
`tokenizer.json`. Pass that directory to `--run-dir`. These are checkpoints for
this repository's PyTorch architecture, rather than Transformers `AutoModel`
checkpoints.

| Model directory under `artifacts/model/models/` | Encoder | Application |
| --- | --- | --- |
| `modar-cnn` | Raster CNN | Full simulated comparison and base-model controls |
| `modar-di` | Peak-list transformer | Public experimental PXRD, CNRS/RRUFF, LiGaCl₄ |
| `modar-cnn-noise-adapted` | Raster CNN | Synthetic-noise evaluation and laboratory powders, including LiCoO₂ |
| `modar-cnn-large-cell` | Raster CNN | Large-cell specialist |
| `modar-cnn-large-cell-noise-adapted` | Raster CNN | Noise-adapted large-cell specialist |
| `modar-mp20` | Raster CNN | Separate MP-20 benchmark |

To obtain another model, repeat the model download with its directory name in
`--include`, for example `'models/modar-di/*'`.

| Input collection | Training | Validation | Test | Local input directory |
| --- | ---: | ---: | ---: | --- |
| Tier-A | 3,456,202 | 34,039 | 33,944 | `artifacts/inputs/tier-a` after materialization |
| MP-20 | 27,136 | 9,047 | 9,046 | `artifacts/inputs/mp20` after materialization |
| CNRS | — | — | 126 | `artifacts/data/benchmarks/cnrs126` |
| RRUFF | — | — | 269 | `artifacts/data/benchmarks/rruff269` |
| Public experimental PXRD | — | — | 337 | `artifacts/data/benchmarks/public337` |
| Laboratory powders | — | — | 10 | `artifacts/data/benchmarks/laboratory10` |

For MP-20, download `'data/mp20/test-*.jsonl.gz'` and materialize with
`--config mp20`. Use `modar-mp20` for this separate protocol. Source provenance
and the MP-20 pretraining-overlap audit are distributed with the dataset.

## Generate candidates

A quick CPU check loads the published weights, constructs the selected inputs,
and verifies a finite forward pass:

```bash
python crystal_language/generate.py \
  --run-dir artifacts/model/models/modar-cnn \
  --data-dir artifacts/inputs/tier-a --split test --index 0 \
  --condition-set pxrd_neutron_xrf --device cpu --check-only
```

Generate four candidates using PXRD, the supplied formula, and the supplied
lattice:

```bash
python crystal_language/generate.py \
  --run-dir artifacts/model/models/modar-cnn \
  --data-dir artifacts/inputs/tier-a --split test --index 0 \
  --condition-set pxrd_formula --use-lattice --constrain formula \
  --num-samples 4 --temperature 0.8 --top-k 20 \
  --max-new-tokens 1024 --seed 42
```

The command prints generated token sequences and parsing results as JSON lines.
It does not perform physical reranking or relaxation; use the evaluation
workflow below for ranked structure-recovery metrics. For a custom record,
`--input-jsonl input.jsonl` (also accepts `.jsonl.gz`) overrides the dataset path.
Generation does not use the reference `target` field.

| `--condition-set` | Supplied inputs | `--constrain` |
| --- | --- | --- |
| `pxrd` | PXRD | `none` |
| `pxrd_neutron` | PXRD + ND | `none` |
| `pxrd_xrf` | PXRD + elemental mass fractions | `none` |
| `pxrd_neutron_xrf` | PXRD + ND + elemental mass fractions | `none` |
| `pxrd_formula` | PXRD + exact chemical formula | `formula` |

Add `--use-lattice` only when lattice parameters are supplied. Omit it for
lattice-free inference. `xrf` is the implementation name for elemental mass
fractions, not a measured fluorescence spectrum.

## Evaluate structure recovery

The following evaluates the complete Tier-A test split using 100 candidates per
query and physical reranking, without supplying a lattice:

```bash
python crystal_language/evaluate_generation.py \
  --run-dir artifacts/model/models/modar-cnn \
  --data-dir artifacts/inputs/tier-a --split test \
  --out-dir results/tier-a/multimodal-free \
  --condition-set pxrd_neutron_xrf --constrain none \
  --max-records 0 --num-samples 100 --samples-per-batch 20 \
  --temperature 0.8 --top-k-sampling 20 --max-new-tokens 1024 \
  --top-ks 1,5,10,20,50 --seed 42 --amp bf16 \
  --rerank --rerank-workers 4 --store-structures
```

Repeat for the five input conditions above, with and without `--use-lattice`.
The formula reference uses both `--condition-set pxrd_formula` and
`--constrain formula`. For a short execution check, use `--max-records 2
--num-samples 4 --top-ks 1 --samples-per-batch 4`; on CPU use `--device cpu
--amp none --rerank-workers 1`.

Evaluation writes a candidate JSONL and a `*_summary.json` under `--out-dir`.
Top-k recovery means that at least one of the highest-ranked k candidates
matches the reference structure. `--start-index` and `--max-records` support
sharding; record the seed, shard boundaries, and sampling settings when
comparing reruns. Sampling and hardware can affect individual candidates.

### Experimental PXRD and refinement

Download `modar-di` and generate the RRUFF candidate pool with formula and
lattice supplied:

```bash
hf download Coooolin/MODAR --revision "$MODAR_MODEL_REV" \
  --include 'models/modar-di/*' --local-dir artifacts/model

python crystal_language/evaluate_generation.py \
  --run-dir artifacts/model/models/modar-di \
  --data-dir artifacts/data/benchmarks/rruff269 --split test \
  --out-dir results/rruff269 \
  --condition-set pxrd_formula --use-lattice --constrain formula \
  --max-records 0 --num-samples 100 --samples-per-batch 20 \
  --temperature 1.0 --top-k-sampling 50 --max-new-tokens 1024 \
  --top-ks 1,20 --seed 42 --amp bf16 \
  --rerank --rerank-workers 4 --store-structures
```

Use `benchmarks/cnrs126` and a separate output directory for CNRS. The published
experimental comparisons additionally use physics–energy fusion and
fixed-lattice relaxation. Generation-only summaries are the pre-refinement
results.

<details>
<summary>Run physics–energy ranking and fixed-lattice refinement</summary>

Install the optional potentials; their model weights are downloaded by the
respective libraries on first use.

```bash
python -m pip install -r requirements-refinement.txt

python scripts/score_candidates_mlip.py \
  --eval-dir results/rruff269 --backend chgnet \
  --out results/rruff269-energies.jsonl

for backend in chgnet mace; do
  python scripts/refine_match_candidates.py \
    --eval-dir results/rruff269 \
    --native artifacts/data/benchmarks/rruff269/test.jsonl \
    --out-dir "results/rruff269-${backend}" --backend "$backend" \
    --rank-by fusion --energy results/rruff269-energies.jsonl \
    --top 20 --max-relax 20 --fmax 0.05 --steps 300
done

python scripts/merge_refined_pools.py \
  --base results/rruff269-chgnet --extra results/rruff269-mace \
  --out results/rruff269-refined

python scripts/rank_eval_pools.py \
  --pool results/rruff269-refined --chgnet results/rruff269-energies.jsonl \
  --rps rruff --out results/rruff269-ranking.json
```

Read the `F_chgnet` ranking for physics–CHGNet reciprocal-rank fusion. Match flags
count a candidate that matches before or after the specified relaxation passes.
The CHGNet and MACE passes keep the lattice fixed; MACE uses `medium-mpa-0`.
The RPS figures printed by this analysis are stored reference counts. Rerunning
RPS itself requires its separate model weights and environment, using
`third_party/RealPXRD-Solver/` and `scripts/run_realpxrd_solver_eval.py`.

</details>

For public-337, use `modar-di` with the selected composition/diffraction
conditions. For the laboratory series, use `modar-cnn-noise-adapted`. Auxiliary
ND and composition in these collections are derived from reference CIFs; their
PXRD is measured. The LiGaCl₄ release provides repeated-run statistics and figure
source data; its raw measured neutron pattern and generated candidate pools are
not included, so that case cannot be rerun from the released files alone.

## Training and data preparation

Download the training and validation shards and restore them in the same input
directory as the test split:

```bash
hf download Coooolin/MODAR-data --repo-type dataset \
  --revision "$MODAR_DATA_REV" \
  --include SPLITS.json --include 'data/tier-a/train-*.jsonl.gz' \
  --include 'data/tier-a/valid-*.jsonl.gz' \
  --local-dir artifacts/data

for split in train valid; do
  python scripts/materialize_dataset.py \
    --dataset-dir artifacts/data --config tier-a --split "$split" \
    --output-dir artifacts/inputs/tier-a
done
```

The uncompressed Tier-A training split occupies about 16.2 GB. An example
single-GPU training configuration is:

```bash
python crystal_language/train.py \
  --arch v2 --rep v2 --spectra-encoder cnn \
  --data-dir artifacts/inputs/tier-a --out-dir results/train-cnn \
  --n-layer 16 --n-head 16 --n-embd 1024 --block-size 1024 \
  --condition-set mix --use-lattice mix --cond-adanorm --coord-value-emb \
  --batch-size 16 --grad-accum-steps 2 --lr 0.0003 \
  --max-steps 54000 --warmup-steps 1000 --amp bf16 --lazy-data always
```

For distributed training, launch with `torchrun`. Match the chosen checkpoint's
`config.json`, including the global batch size (per-device batch × accumulation
× world size), when reproducing its training setup. Peak-list and specialist
models use the corresponding initialization/fine-tuning settings; `--init-from`
accepts a checkpoint file. The released checkpoints contain model weights
and configuration, without optimizer state.

Each model-facing JSONL record contains `cond`, `spectra`, and reference `target`
tokens. Peak positions in `spectra` are 2θ degrees, normally at 1.54184 Å over
5–80°. The peak-list encoder converts them to d-spacings; the CNN uses a
1,501-point raster. Only the selected condition blocks and spectra are inference
inputs. Exact formula and lattice are supplied only when their flags are enabled.

The ten `XRD/*.raw`–`*.cif` pairs are the laboratory inputs. To rebuild their
prepared records:

```bash
python data_prep/official_multisource/scripts/build_collaborator_exp_pxrd.py \
  --xrd-dir XRD --out-dir artifacts/laboratory-rebuilt
```

The source-download, LMDB construction, multimodal enrichment, deduplication,
and export utilities are in `data_prep/official_multisource/`. These are useful
for rebuilding from source databases; the frozen Hugging Face splits are the
input for the evaluations above. Source-data rebuilding may additionally need
`lmdb`, `requests`, and `pandas`, plus source-specific access credentials.

## Repository layout

```text
crystal_language/                model, tokenizer, training, generation, evaluation
scripts/                        materialization, ranking, refinement, analysis, job launchers
data_prep/official_multisource/  source-data builders and exporters
XRD/                            ten paired laboratory PXRD and reference-CIF inputs
third_party/RealPXRD-Solver/     reference implementation and its license
```

The Slurm launchers in `scripts/` provide cluster job examples; adapt their
scheduler and filesystem settings to your environment. Model weights,
benchmark inputs, all 21 supplementary tables, and figure source numbers are
hosted in the linked Hugging Face releases.

## License and citation

The project code is distributed under the [MIT license](LICENSE). Third-party
code retains its own license. Model and dataset reuse conditions and original
source attribution are described in their Hugging Face repositories.

If you use MODAR, cite *An optional-modality autoregressive language for
crystal-structure solution from complementary X-ray and neutron probes* and the
model/data revisions used. A publication identifier will be added when available.
