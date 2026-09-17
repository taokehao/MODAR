#!/bin/bash
# Big-cell noisy eval with noisy-aware rerank defaults (σ=0.40 ±0.20°).
# Override USE_LATTICE / OUT_DIR per submission.
#SBATCH --job-name=noisy_big_rr
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/noisy_bigcell_rr_%j.out
#SBATCH --error=logs/noisy_bigcell_rr_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_bigcell_noisy_exp}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/crystal_language/eval_results/bigcell_ab/data}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/bigcell_ab/noisy_bigcell_free_rraware}
SPLIT=${SPLIT:-test_big20}
COND=${COND:-pxrd_neutron_xrf}
USE_LATTICE=${USE_LATTICE:-0}
MAX_RECORDS=${MAX_RECORDS:-500}
NUM_SAMPLES=${NUM_SAMPLES:-100}
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-20}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TOP_KS=${TOP_KS:-1,5,10,20,50}
RERANK_WORKERS=${RERANK_WORKERS:-4}
AMP=${AMP:-bf16}
NOISE_PROFILE=${NOISE_PROFILE:-experimental}
NOISE_SEED=${NOISE_SEED:-20260706}

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

USE_LATTICE_ARG=()
if [ "${USE_LATTICE}" = "1" ]; then
  USE_LATTICE_ARG=(--use-lattice)
fi

echo "BIGCELL-noisy-rraware run=${RUN_DIR} out=${OUT_DIR} lat=${USE_LATTICE} $(date)"

srun python -u crystal_language/evaluate_generation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --condition-set "${COND}" \
  "${USE_LATTICE_ARG[@]}" \
  --rerank --rerank-workers "${RERANK_WORKERS}" \
  --target-mode full \
  --constrain none \
  --amp "${AMP}" \
  --noise-profile "${NOISE_PROFILE}" \
  --noise-seed "${NOISE_SEED}" \
  --store-structures \
  --start-index 0 \
  --max-records "${MAX_RECORDS}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo "DONE $(date)"
