#!/bin/bash
# ReST / RFT stage-1 fine-tune: continue the Tier-A S2 model on the distilled
# (model-correct) dataset built by build_rft_dataset.py.
#
# Key points:
#   * --init-from the base checkpoint (fresh optimizer + short cosine schedule).
#   * The base run's tokenizer.json is copied into the new out_dir FIRST so the
#     vocab matches the checkpoint (train.py requires this for --init-from).
#   * The distilled data dir reuses the original valid.jsonl (symlinked) so the
#     validation curve is comparable to pretraining.
#   * Low LR + few steps to sharpen the policy without forgetting.
#
#SBATCH --job-name=crylm_rft
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4090D:2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/crystal_lm_rft_%j.out
#SBATCH --error=logs/crystal_lm_rft_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-2}
BASE_RUN=${BASE_RUN:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
INIT_CKPT=${INIT_CKPT:-${BASE_RUN}/checkpoint_last.pt}
BASE_DATA=${BASE_DATA:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_tierA_rft1}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_rft1}

BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
# Distilled targets are lattice-free solutions; train under lattice-free prompts
# only (use-lattice=never) to avoid the known-lattice prompt/target cell mismatch
# that regressed known-lattice in the first ReST attempt. Override as needed.
USE_LATTICE=${USE_LATTICE:-mix}
LR=${LR:-5e-5}
MAX_STEPS=${MAX_STEPS:-600}
WARMUP_STEPS=${WARMUP_STEPS:-50}
EVAL_EVERY=${EVAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-200}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"

# --- vocab: reuse base tokenizer so --init-from weights load (strict) ---
if [[ ! -f "${RUN_DIR}/tokenizer.json" ]]; then
  cp "${BASE_RUN}/tokenizer.json" "${RUN_DIR}/tokenizer.json"
fi

# --- data: distilled train.jsonl must exist; reuse original valid/test ---
if [[ ! -s "${DATA_DIR}/train.jsonl" ]]; then
  echo "FATAL: missing distilled ${DATA_DIR}/train.jsonl (run build_rft_dataset.py first)" >&2
  exit 1
fi
for split in valid test; do
  if [[ ! -e "${DATA_DIR}/${split}.jsonl" ]]; then
    ln -s "${BASE_DATA}/${split}.jsonl" "${DATA_DIR}/${split}.jsonl"
  fi
done
ls -lh "${DATA_DIR}"/*.jsonl

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} init_from=${INIT_CKPT} data=${DATA_DIR} run=${RUN_DIR}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

COMMON_ARGS=(
  --arch v2 --rep v2
  --data-dir "${DATA_DIR}"
  --out-dir "${RUN_DIR}"
  --condition-set mix
  --use-lattice "${USE_LATTICE}"
  --target-mode full
  --block-size "${BLOCK_SIZE}"
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}"
  --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}"
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}"
  --eval-every "${EVAL_EVERY}" --save-every "${SAVE_EVERY}"
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
  --cond-adanorm --cond-drop "${COND_DROP}"
  --coord-value-emb
  --lazy-data never
  --init-from "${INIT_CKPT}"
)

echo "=== prepare: tokenizer + index $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
