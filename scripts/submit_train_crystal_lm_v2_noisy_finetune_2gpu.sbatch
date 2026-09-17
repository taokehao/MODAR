#!/bin/bash
#SBATCH --job-name=crylm_noisyft
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4090D:2
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/crystal_lm_noisyft_%j.out
#SBATCH --error=logs/crystal_lm_noisyft_%j.err

# Two-GPU robust fine-tuning with online experimental-noise augmentation.
# Use MIN_SITES=21 to fine-tune the large-cell specialist; leave empty for the
# full-distribution base model.  The source checkpoint tokenizer is copied into
# RUN_DIR before --init-from, as required by train.py.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-2}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
BASE_RUN=${BASE_RUN:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
INIT_FROM=${INIT_FROM:-${BASE_RUN}/checkpoint_last.pt}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_noisy_exp}
MIN_SITES=${MIN_SITES:-}

CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-5e-5}
MAX_STEPS=${MAX_STEPS:-6000}
WARMUP_STEPS=${WARMUP_STEPS:-300}
EVAL_EVERY=${EVAL_EVERY:-1000}
SAVE_EVERY=${SAVE_EVERY:-3000}
NUM_WORKERS=${NUM_WORKERS:-6}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}
NOISE_PROFILE=${NOISE_PROFILE:-experimental}
LATTICE_FREE_PROB=${LATTICE_FREE_PROB:-0.5}
MIX_CONDITION_WEIGHTS=${MIX_CONDITION_WEIGHTS:-}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

for split in train valid test; do
  if [[ ! -s "${DATA_DIR}/${split}.jsonl" ]]; then
    echo "FATAL: missing ${DATA_DIR}/${split}.jsonl" >&2
    exit 1
  fi
done
if [[ ! -s "${INIT_FROM}" ]]; then
  echo "FATAL: init checkpoint missing: ${INIT_FROM}" >&2
  exit 1
fi
if [[ ! -s "${RUN_DIR}/tokenizer.json" ]]; then
  cp "$(dirname "${INIT_FROM}")/tokenizer.json" "${RUN_DIR}/tokenizer.json"
  echo "copied tokenizer.json from $(dirname "${INIT_FROM}") into ${RUN_DIR}"
fi

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} run_dir=${RUN_DIR} init=${INIT_FROM} min_sites=${MIN_SITES:-none} noise=${NOISE_PROFILE}"
echo "sampling use_lattice=${USE_LATTICE} lattice_free_prob=${LATTICE_FREE_PROB} mix_condition_weights=${MIX_CONDITION_WEIGHTS:-uniform}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

COMMON_ARGS=(
  --arch v2 --rep v2
  --data-dir "${DATA_DIR}"
  --out-dir "${RUN_DIR}"
  --init-from "${INIT_FROM}"
  --condition-set "${CONDITION_SET}"
  --use-lattice "${USE_LATTICE}"
  --target-mode "${TARGET_MODE}"
  --block-size "${BLOCK_SIZE}"
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}"
  --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}"
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}"
  --eval-every "${EVAL_EVERY}" --save-every "${SAVE_EVERY}"
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
  --cond-adanorm --cond-drop "${COND_DROP}"
  --coord-value-emb
  --noise-profile "${NOISE_PROFILE}"
  --lattice-free-prob "${LATTICE_FREE_PROB}"
  --lazy-data auto
)
if [[ -n "${MIN_SITES}" ]]; then
  COMMON_ARGS+=(--min-sites "${MIN_SITES}")
fi
if [[ -n "${MIX_CONDITION_WEIGHTS}" ]]; then
  COMMON_ARGS+=(--mix-condition-weights "${MIX_CONDITION_WEIGHTS}")
fi

echo "=== prepare: tokenizer + lazy index caches $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
