#!/bin/bash
# Big-cell (>20 atoms) A/B eval: run one checkpoint over the >20 test subset,
# pxrd_neutron_xrf, NO rerank (pure sampling Top-k). USE_LATTICE=1 evaluates
# the known-lattice mode. Submit once per checkpoint x lattice mode (override
# RUN_DIR + OUT_DIR + USE_LATTICE), then compare the summaries.
#
#SBATCH --job-name=crylm_bigab
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/bigcell_ab_%j.out
#SBATCH --error=logs/bigcell_ab_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/crystal_language/eval_results/bigcell_ab/data}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/bigcell_ab/base}
SPLIT=${SPLIT:-test_big20}
COND=${COND:-pxrd_neutron_xrf}
USE_LATTICE=${USE_LATTICE:-0}
MAX_RECORDS=${MAX_RECORDS:-500}
NUM_SAMPLES=${NUM_SAMPLES:-20}
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-20}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TOP_KS=${TOP_KS:-1,5,10,20}
# RERANK=1 turns on physical-consistency rerank (paper main-table protocol,
# usually with NUM_SAMPLES=100). Default 0 = pure sampling Top-k (RL metric).
RERANK=${RERANK:-0}
AMP=${AMP:-bf16}
NOISE_PROFILE=${NOISE_PROFILE:-none}
NOISE_SEED=${NOISE_SEED:-20260706}

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "BIGCELL-AB run=${RUN_DIR} ckpt=${CHECKPOINT} out=${OUT_DIR} cond=${COND} lat=${USE_LATTICE} n=${MAX_RECORDS} $(date)"

USE_LATTICE_ARG=()
if [ "${USE_LATTICE}" = "1" ]; then
  USE_LATTICE_ARG=(--use-lattice)
fi

CONSTRAIN=none
if [ "${COND}" = "pxrd_formula" ]; then
  CONSTRAIN=formula
fi

# Default: NO --rerank (pure sampling Top-k, the metric RL should move).
# RERANK=1 adds physical-consistency rerank for the paper main-table protocol.
RERANK_ARG=()
if [ "${RERANK}" = "1" ]; then
  RERANK_ARG=(--rerank)
fi

srun python -u crystal_language/evaluate_generation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --condition-set "${COND}" \
  "${USE_LATTICE_ARG[@]}" \
  "${RERANK_ARG[@]}" \
  --target-mode full \
  --constrain "${CONSTRAIN}" \
  --amp "${AMP}" \
  --noise-profile "${NOISE_PROFILE}" \
  --noise-seed "${NOISE_SEED}" \
  --start-index 0 \
  --max-records "${MAX_RECORDS}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo "DONE $(date)"
