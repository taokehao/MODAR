#!/bin/bash
#SBATCH --job-name=crylm_v2_mp20
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/crystal_lm_v2_mp20_%j.out
#SBATCH --error=logs/crystal_lm_v2_mp20_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_mp20_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp20_v2_s1_block1024}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=8

srun python -u crystal_language/train.py \
  --arch v2 --rep v2 \
  --data-dir "${DATA_DIR}" \
  --out-dir "${RUN_DIR}" \
  --condition-set "${CONDITION_SET:-mix}" \
  --use-lattice "${USE_LATTICE:-mix}" \
  --target-mode "${TARGET_MODE:-full}" \
  --block-size "${BLOCK_SIZE:-1024}" \
  --n-layer "${N_LAYER:-12}" --n-head "${N_HEAD:-12}" --n-embd "${N_EMBD:-768}" \
  --batch-size "${BATCH_SIZE:-32}" --grad-accum-steps "${GRAD_ACCUM:-1}" \
  --lr "${LR:-3e-4}" --max-steps "${MAX_STEPS:-12000}" --warmup-steps "${WARMUP_STEPS:-500}" \
  --num-workers "${NUM_WORKERS:-6}" --amp "${AMP:-bf16}"
