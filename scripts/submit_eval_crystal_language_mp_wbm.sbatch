#!/bin/bash
#SBATCH --job-name=crylm_eval
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/crystal_language_eval_%j.out
#SBATCH --error=logs/crystal_language_eval_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp_wbm_v1_block2048}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_mp_wbm}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/mp_wbm_v1_block2048}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}

SPLIT=${SPLIT:-valid}
CONDITION_SET=${CONDITION_SET:-pxrd_formula}
USE_LATTICE=${USE_LATTICE:-true}
TARGET_MODE=${TARGET_MODE:-full}
INJECT_KNOWN_LATTICE=${INJECT_KNOWN_LATTICE:-false}
START_INDEX=${START_INDEX:-0}
MAX_RECORDS=${MAX_RECORDS:-500}
NUM_SAMPLES=${NUM_SAMPLES:-50}
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-10}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TOP_KS=${TOP_KS:-1,20,50}
CONSTRAIN=${CONSTRAIN:-none}
RERANK=${RERANK:-false}
RERANK_WORKERS=${RERANK_WORKERS:-6}
AMP=${AMP:-bf16}
STORE_STRUCTURES=${STORE_STRUCTURES:-true}

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "============================================================"
echo "Job:            ${SLURM_JOB_ID:-manual}"
echo "Node:           ${SLURM_JOB_NODELIST:-local}"
echo "Run dir:        ${RUN_DIR}"
echo "Checkpoint:     ${CHECKPOINT}"
echo "Data dir:       ${DATA_DIR}"
echo "Out dir:        ${OUT_DIR}"
echo "Split:          ${SPLIT}"
echo "Condition set:  ${CONDITION_SET}"
echo "Use lattice:    ${USE_LATTICE}"
echo "Target mode:    ${TARGET_MODE}"
echo "Inject lattice: ${INJECT_KNOWN_LATTICE}"
echo "Start index:    ${START_INDEX}"
echo "Max records:    ${MAX_RECORDS}"
echo "Num samples:    ${NUM_SAMPLES}"
echo "Python:         $(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"
echo "Time:           $(date)"
echo "============================================================"

USE_LATTICE_ARG=()
if [ "${USE_LATTICE}" = "true" ]; then
  USE_LATTICE_ARG=(--use-lattice)
fi

INJECT_KNOWN_LATTICE_ARG=()
if [ "${INJECT_KNOWN_LATTICE}" = "true" ]; then
  INJECT_KNOWN_LATTICE_ARG=(--inject-known-lattice)
fi

RERANK_ARG=()
if [ "${RERANK}" = "true" ]; then
  RERANK_ARG=(--rerank --rerank-workers "${RERANK_WORKERS}")
fi

STORE_STRUCTURES_ARG=()
if [ "${STORE_STRUCTURES}" = "true" ]; then
  STORE_STRUCTURES_ARG=(--store-structures)
fi

LOG_FILE="${LOG_DIR}/crystal_language_eval_${SLURM_JOB_ID:-manual}.log"

srun python -u crystal_language/evaluate_generation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --condition-set "${CONDITION_SET}" \
  "${USE_LATTICE_ARG[@]}" \
  --target-mode "${TARGET_MODE}" \
  "${INJECT_KNOWN_LATTICE_ARG[@]}" \
  --constrain "${CONSTRAIN}" \
  "${RERANK_ARG[@]}" \
  --amp "${AMP}" \
  "${STORE_STRUCTURES_ARG[@]}" \
  --start-index "${START_INDEX}" \
  --max-records "${MAX_RECORDS}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee -a "${LOG_FILE}"

exit ${PIPESTATUS[0]}
