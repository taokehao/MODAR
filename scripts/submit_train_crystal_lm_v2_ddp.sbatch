#!/bin/bash
#SBATCH --job-name=crylm_v2
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4090D:6
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/crystal_lm_v2_ddp_%j.out
#SBATCH --error=logs/crystal_lm_v2_ddp_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-6}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_mp_wbm_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp_wbm_v2_s1_block1024}

CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-12}
N_HEAD=${N_HEAD:-12}
N_EMBD=${N_EMBD:-768}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-3e-4}
MAX_STEPS=${MAX_STEPS:-12000}
WARMUP_STEPS=${WARMUP_STEPS:-500}
NUM_WORKERS=${NUM_WORKERS:-6}
AMP=${AMP:-bf16}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} run_dir=${RUN_DIR} data=${DATA_DIR}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py \
  --arch v2 --rep v2 \
  --data-dir "${DATA_DIR}" \
  --out-dir "${RUN_DIR}" \
  --condition-set "${CONDITION_SET}" \
  --use-lattice "${USE_LATTICE}" \
  --target-mode "${TARGET_MODE}" \
  --block-size "${BLOCK_SIZE}" \
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}" \
  --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}" \
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}" \
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
