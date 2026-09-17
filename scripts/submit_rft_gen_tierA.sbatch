#!/bin/bash
# ReST / RFT stage-1: mine model-correct samples on a TRAIN subset.
#
# Runs evaluate_generation.py on the Tier-A TRAIN split for ONE condition rung
# (default pxrd_neutron_xrf, lattice-free), sampling K candidates per record and
# labelling each with StructureMatcher vs ground truth. No rerank (we have GT, and
# the physical-consistency simulation is the expensive part). --store-structures
# keeps each candidate's parsed structure so build_rft_dataset.py can re-encode the
# correct ones into distilled v2 targets.
#
# Array layout: NUM_SHARDS tasks (one lattice mode, one rung) covering
# NUM_SHARDS*SHARD_SIZE train records.
#
#SBATCH --job-name=crylm_rftgen
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/rft_gen_%A_%a.out
#SBATCH --error=logs/rft_gen_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/rft_gen_tierA_train}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}

SPLIT=${SPLIT:-train}
COND=${COND:-pxrd_neutron_xrf}
LAT=${LAT:-0}
SHARD_SIZE=${SHARD_SIZE:-500}
NUM_SAMPLES=${NUM_SAMPLES:-32}
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-16}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TOP_KS=${TOP_KS:-1,10,50}
AMP=${AMP:-bf16}

TID=${SLURM_ARRAY_TASK_ID:-0}
START_INDEX=$(( TID * SHARD_SIZE ))
MAX_RECORDS=${MAX_RECORDS:-${SHARD_SIZE}}

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
echo "RFT-GEN ${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Condition:  ${COND}   Lattice: ${LAT}   Split: ${SPLIT}"
echo "Shard start=${START_INDEX} max=${MAX_RECORDS}  num_samples=${NUM_SAMPLES}"
echo "Run dir: ${RUN_DIR}   Out dir: ${OUT_DIR}"
echo "Time: $(date)"
echo "============================================================"

# NOTE: no --rerank (train split has GT; structure_match labels candidates).
srun python -u crystal_language/evaluate_generation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --condition-set "${COND}" \
  "${USE_LATTICE_ARG[@]}" \
  --target-mode full \
  --constrain none \
  --amp "${AMP}" \
  --store-structures \
  --start-index "${START_INDEX}" \
  --max-records "${MAX_RECORDS}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
