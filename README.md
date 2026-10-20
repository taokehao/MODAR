# MODAR

**M**ultimodal **O**ptional-prefix **D**iffraction **A**utoregressive **R**etrieval

One autoregressive crystal language that treats PXRD, neutron diffraction, and
elemental mass fractions as droppable prefixes. This repository is the code,
inference/evaluation scripts, figure packages, and manuscript drafts for that
work.

Checkpoints, sequence corpora, and candidate pools are too large for GitHub
and will be published separately on Hugging Face.

## Layout

```text
crystal_language/     model, tokenizer, generate, train, Top-k eval
scripts/              Slurm + local inference / refine / RPS comparison
data_prep/official_multisource/   data-build scripts (not the raw dumps)
```

## Inference

```bash
python crystal_language/generate.py \
  --run-dir /path/to/checkpoint_dir \
  --data-dir /path/to/jsonl \
  --split test --index 0 \
  --condition-set pxrd_xrf --use-lattice \
  --num-samples 4
```

Paper protocol (N=100, rerank, optional formula constraint):

```bash
python crystal_language/evaluate_generation.py \
  --run-dir /path/to/checkpoint_dir \
  --data-dir /path/to/jsonl \
  --split test --condition-set pxrd_neutron_xrf \
  --use-lattice --rerank --num-samples 100 \
  --constrain none --amp bf16 --store-structures
```

Cluster array jobs live in `scripts/submit_eval_ladder_*.sbatch`.

## Training

```bash
python crystal_language/train.py \
  --arch v2 --rep v2 \
  --data-dir data_prep/outputs/crystal_language_unified_tierA_v2 \
  --out-dir crystal_language/runs/example \
  --condition-set mix --use-lattice mix \
  --spectra-encoder di --cond-adanorm --coord-value-emb \
  --noise-profile experimental
```
