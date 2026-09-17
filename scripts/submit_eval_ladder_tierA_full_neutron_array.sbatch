#!/bin/bash
# Full Tier-A test evaluation for the neutron-focused multimodal ladder.
#
# Rungs:
#   pxrd -> pxrd+neutron -> pxrd+xrf -> pxrd+neutron+xrf
#        -> pxrd+neutron+xrf+xps -> pxrd+formula(oracle)
#
# Array layout (NUM_SHARDS shards x 2 lattice modes x 6 condition sets):
#   shard    = TID % NUM_SHARDS
#   lat      = (TID / NUM_SHARDS) % 2          # 0 lattice-free, 1 known lattice
#   cond_idx = (TID / NUM_SHARDS) / 2          # 0..5 into CONDS
# Default NUM_SHARDS=68, SHARD_SIZE=500 covers all 33,944 Tier-A test records.
#
#SBATCH --job-name=crylm_neuFull
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-815
#SBATCH --output=logs/ladder_tierA_full_neutron_%A_%a.out
#SBATCH --error=logs/ladder_tierA_full_neutron_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/ladder_tierA_s2_full_neutron}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}

SPLIT=${SPLIT:-test}
NUM_SHARDS=${NUM_SHARDS:-68}
SHARD_SIZE=${SHARD_SIZE:-500}
NUM_SAMPLES=${NUM_SAMPLES:-100}
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-20}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TOP_KS=${TOP_KS:-1,10,50}
RERANK_WORKERS=${RERANK_WORKERS:-4}
AMP=${AMP:-bf16}
NOISE_PROFILE=${NOISE_PROFILE:-none}
NOISE_SEED=${NOISE_SEED:-20260706}

# Neutron-focused core ladder, low -> high information.
CONDS=(pxrd pxrd_neutron pxrd_xrf pxrd_neutron_xrf pxrd_neutron_xrf_xps pxrd_formula)

TID=${SLURM_ARRAY_TASK_ID:-0}
SHARD=$(( TID % NUM_SHARDS ))
REST=$(( TID / NUM_SHARDS ))
LAT=$(( REST % 2 ))
COND_IDX=$(( REST / 2 ))
COND=${CONDS[$COND_IDX]}
START_INDEX=$(( SHARD * SHARD_SIZE ))
MAX_RECORDS=${MAX_RECORDS:-${SHARD_SIZE}}

# Only the oracle formula rung gets formula-constrained decoding.
CONSTRAIN=none
if [ "${COND}" = "pxrd_formula" ]; then
  CONSTRAIN=formula
fi

USE_LATTICE_ARG=()
if [ "${LAT}" = "1" ]; then
  USE_LATTICE_ARG=(--use-lattice)
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "============================================================"
echo "Job:            ${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:           ${SLURM_JOB_NODELIST:-local}"
echo "Condition set:  ${COND}  (rung ${COND_IDX}/5)"
echo "Lattice:        ${LAT} (0=free, 1=known)"
echo "Constrain:      ${CONSTRAIN}"
echo "Shard:          ${SHARD}/${NUM_SHARDS}  start=${START_INDEX}  max=${MAX_RECORDS}"
echo "Split:          ${SPLIT}"
echo "Num samples:    ${NUM_SAMPLES}  (batch ${SAMPLES_PER_BATCH})"
echo "Run dir:        ${RUN_DIR}"
echo "Out dir:        ${OUT_DIR}"
echo "Python:         $(command -v python)"
echo "Time:           $(date)"
echo "============================================================"

srun python -u crystal_language/evaluate_generation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --condition-set "${COND}" \
  "${USE_LATTICE_ARG[@]}" \
  --target-mode full \
  --constrain "${CONSTRAIN}" \
  --rerank --rerank-workers "${RERANK_WORKERS}" \
  --amp "${AMP}" \
  --noise-profile "${NOISE_PROFILE}" \
  --noise-seed "${NOISE_SEED}" \
  --store-structures \
  --start-index "${START_INDEX}" \
  --max-records "${MAX_RECORDS}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
