#!/bin/bash
#SBATCH --job-name=crylm_mev
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/crystal_language_medium_eval_%A_%a.out
#SBATCH --error=logs/crystal_language_medium_eval_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
SHARD_SIZE=${SHARD_SIZE:-50}
START_INDEX=$(( SLURM_ARRAY_TASK_ID * SHARD_SIZE ))

export RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp_wbm_medium_block2048}
export OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/eval_results/mp_wbm_medium_block2048}
export CONDITION_SET=${CONDITION_SET:-pxrd_formula}
export USE_LATTICE=${USE_LATTICE:-true}
export MAX_RECORDS=${MAX_RECORDS:-${SHARD_SIZE}}
export START_INDEX
export NUM_SAMPLES=${NUM_SAMPLES:-50}
export SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-10}
export TEMPERATURE=${TEMPERATURE:-0.8}
export TOP_K_SAMPLING=${TOP_K_SAMPLING:-20}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

cd "${PROJECT_DIR}"

bash scripts/submit_eval_crystal_language_mp_wbm.sbatch
