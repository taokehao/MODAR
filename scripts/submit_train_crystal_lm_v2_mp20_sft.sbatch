#!/bin/bash
#SBATCH --job-name=crylm_mp20sft
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/crystal_lm_v2_mp20_sft_%j.out
#SBATCH --error=logs/crystal_lm_v2_mp20_sft_%j.err

# MP-20 SFT from the RealPXRD 6M pretrained checkpoint (A6 follow-up).
# Reuses the pretrain tokenizer (MP-20 vocab verified to be a subset) so the
# checkpoint weights load strictly.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_mp20_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp20_v2_s1_sft_realpxrd}
PRETRAIN_RUN=${PRETRAIN_RUN:-${PROJECT_DIR}/crystal_language/runs/realpxrd_v2_s1_ada_cve}
INIT_FROM=${INIT_FROM:-${PRETRAIN_RUN}/checkpoint_last.pt}

CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-12}
N_HEAD=${N_HEAD:-12}
N_EMBD=${N_EMBD:-768}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-3e-5}
MAX_STEPS=${MAX_STEPS:-6000}
WARMUP_STEPS=${WARMUP_STEPS:-200}
EVAL_EVERY=${EVAL_EVERY:-250}
SAVE_EVERY=${SAVE_EVERY:-500}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Reuse the pretrain vocab so --init-from loads strictly.
cp -n "${PRETRAIN_RUN}/tokenizer.json" "${RUN_DIR}/tokenizer.json"

echo "node=$(hostname) run_dir=${RUN_DIR} init_from=${INIT_FROM}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

srun python -u crystal_language/train.py \
  --arch v2 --rep v2 \
  --data-dir "${DATA_DIR}" \
  --out-dir "${RUN_DIR}" \
  --init-from "${INIT_FROM}" \
  --condition-set "${CONDITION_SET}" \
  --use-lattice "${USE_LATTICE}" \
  --target-mode "${TARGET_MODE}" \
  --block-size "${BLOCK_SIZE}" \
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}" \
  --eval-every "${EVAL_EVERY}" --save-every "${SAVE_EVERY}" \
  --num-workers "${NUM_WORKERS}" --amp "${AMP}" \
  --cond-adanorm --cond-drop "${COND_DROP}" \
  --coord-value-emb

echo "DONE $(date)"
