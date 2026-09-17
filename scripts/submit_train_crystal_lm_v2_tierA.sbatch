#!/bin/bash
#SBATCH --job-name=crylm_tierA
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4090D:6
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/crystal_lm_v2_tierA_%j.out
#SBATCH --error=logs/crystal_lm_v2_tierA_%j.err

# Stage B: Tier-A pretraining on the self-curated, deduped unified corpus
# (tight cell-metric dedup -> tier_a keep list, ~3.46M train sequences).
# S2 (16L/1024d/16H) with AdaLN conditioning + coordinate value embeddings.
# A single-process --prepare-only pass builds the tokenizer + lazy index caches
# before torchrun starts.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-6}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
INIT_FROM=${INIT_FROM:-}

CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
# S2 scale (plan section 2.1).
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
# Effective batch 16*2*6 = 192 (matches the validated V2 recipe); batch 16 keeps
# the 220M model comfortably within 24 GB.
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-3e-4}
# ~3 epochs over 3.456M train: 3*3.456M / 192 ~= 54000 steps.
MAX_STEPS=${MAX_STEPS:-54000}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}
NOISE_PROFILE=${NOISE_PROFILE:-none}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

for split in train valid test; do
  if [[ ! -s "${DATA_DIR}/${split}.jsonl" ]]; then
    echo "FATAL: missing ${DATA_DIR}/${split}.jsonl (seq export incomplete?)" >&2
    exit 1
  fi
done
if [[ -n "${INIT_FROM}" && ! -s "${INIT_FROM}" ]]; then
  echo "FATAL: init checkpoint missing: ${INIT_FROM}" >&2
  exit 1
fi
if [[ -n "${INIT_FROM}" && ! -s "${RUN_DIR}/tokenizer.json" ]]; then
  cp "$(dirname "${INIT_FROM}")/tokenizer.json" "${RUN_DIR}/tokenizer.json"
  echo "copied tokenizer.json from $(dirname "${INIT_FROM}") into ${RUN_DIR}"
fi
ls -lh "${DATA_DIR}"/*.jsonl

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} run_dir=${RUN_DIR} data=${DATA_DIR}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

COMMON_ARGS=(
  --arch v2 --rep v2
  --data-dir "${DATA_DIR}"
  --out-dir "${RUN_DIR}"
  --condition-set "${CONDITION_SET}"
  --use-lattice "${USE_LATTICE}"
  --target-mode "${TARGET_MODE}"
  --block-size "${BLOCK_SIZE}"
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}"
  --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}"
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}"
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
  --cond-adanorm --cond-drop "${COND_DROP}"
  --coord-value-emb
  --noise-profile "${NOISE_PROFILE}"
  --lazy-data auto
)
if [[ -n "${INIT_FROM}" ]]; then
  COMMON_ARGS+=(--init-from "${INIT_FROM}")
fi

echo "=== prepare: tokenizer + lazy index caches $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
