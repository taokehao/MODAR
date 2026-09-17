#!/bin/bash
# ReST stage-1 A/B eval: compare a checkpoint's NO-RERANK Top-k on a small Tier-A
# test subset across a few key rungs x {lattice-free, known-lattice}.
#
# No --rerank: we want pure sampling Top-1 (the metric ReST should move), and it
# also skips the expensive pymatgen simulation so the A/B is fast/cheap.
# Run once per checkpoint (override RUN_DIR + OUT_DIR), then merge_ladder each.
#
#SBATCH --job-name=crylm_rftcmp
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/rft_cmp_%A_%a.out
#SBATCH --error=logs/rft_cmp_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/rft_compare_base}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}

SPLIT=${SPLIT:-test}
# space-separated rung list (default: mined rung + composition control + oracle)
CONDS_STR=${CONDS_STR:-"pxrd_neutron_xrf pxrd_xrf pxrd_formula"}
read -r -a CONDS <<< "${CONDS_STR}"
NUM_SHARDS=${NUM_SHARDS:-2}
SHARD_SIZE=${SHARD_SIZE:-500}
NUM_SAMPLES=${NUM_SAMPLES:-50}
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-25}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TOP_KS=${TOP_KS:-1,10,50}
AMP=${AMP:-bf16}

TID=${SLURM_ARRAY_TASK_ID:-0}
SHARD=$(( TID % NUM_SHARDS ))
REST=$(( TID / NUM_SHARDS ))
LAT=$(( REST % 2 ))
COND_IDX=$(( REST / 2 ))
COND=${CONDS[$COND_IDX]}
START_INDEX=$(( SHARD * SHARD_SIZE ))
MAX_RECORDS=${MAX_RECORDS:-${SHARD_SIZE}}

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

echo "RFT-CMP ${SLURM_ARRAY_JOB_ID:-manual}_${TID}  cond=${COND} lat=${LAT} shard=${SHARD} start=${START_INDEX} run=${RUN_DIR}  $(date)"

# NO --rerank: pure sampling Top-k (this is the ReST metric of interest).
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
  --amp "${AMP}" \
  --start-index "${START_INDEX}" \
  --max-records "${MAX_RECORDS}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
