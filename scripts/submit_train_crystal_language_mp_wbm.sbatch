#!/bin/bash
#SBATCH --job-name=crylm_mpw
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/crystal_language_mp_wbm_%j.out
#SBATCH --error=logs/crystal_language_mp_wbm_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_mp_wbm}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp_wbm_v1_block2048}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-2048}
MODEL_TYPE=${MODEL_TYPE:-decoder}
N_LAYER=${N_LAYER:-6}
N_ENCODER_LAYER=${N_ENCODER_LAYER:-0}
N_HEAD=${N_HEAD:-6}
N_EMBD=${N_EMBD:-384}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-50000}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-0.0003}
AMP=${AMP:-none}

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "============================================================"
echo "Job:           ${SLURM_JOB_ID:-manual}"
echo "Node:          ${SLURM_JOB_NODELIST:-local}"
echo "Project:       ${PROJECT_DIR}"
echo "Data:          ${DATA_DIR}"
echo "Run dir:       ${RUN_DIR}"
echo "Conda env:     ${CONDA_ENV}"
echo "Condition set: ${CONDITION_SET}"
echo "Use lattice:   ${USE_LATTICE}"
echo "Target mode:   ${TARGET_MODE}"
echo "Block size:    ${BLOCK_SIZE}"
echo "Model type:    ${MODEL_TYPE}"
echo "Model:         layers=${N_LAYER} heads=${N_HEAD} emb=${N_EMBD}"
echo "Encoder layer: ${N_ENCODER_LAYER}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Grad accum:    ${GRAD_ACCUM_STEPS}"
echo "Max steps:     ${MAX_STEPS}"
echo "AMP:           ${AMP}"
echo "Python:        $(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"
echo "Time:          $(date)"
echo "============================================================"

LOG_FILE="${LOG_DIR}/crystal_language_mp_wbm_${SLURM_JOB_ID:-manual}.log"

srun python -u crystal_language/train.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${RUN_DIR}" \
  --condition-set "${CONDITION_SET}" \
  --use-lattice "${USE_LATTICE}" \
  --target-mode "${TARGET_MODE}" \
  --block-size "${BLOCK_SIZE}" \
  --model-type "${MODEL_TYPE}" \
  --n-layer "${N_LAYER}" \
  --n-encoder-layer "${N_ENCODER_LAYER}" \
  --n-head "${N_HEAD}" \
  --n-embd "${N_EMBD}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --amp "${AMP}" \
  2>&1 | tee -a "${LOG_FILE}"

exit ${PIPESTATUS[0]}
