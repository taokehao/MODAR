#!/bin/bash
#SBATCH --job-name=crylm_site
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/crystal_language_sites_medium_%j.out
#SBATCH --error=logs/crystal_language_sites_medium_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
export RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/mp_wbm_sites_medium_block4096}
export LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
export CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

export CONDITION_SET=${CONDITION_SET:-pxrd_formula}
export USE_LATTICE=${USE_LATTICE:-always}
export TARGET_MODE=${TARGET_MODE:-sites}
export BLOCK_SIZE=${BLOCK_SIZE:-4096}
export N_LAYER=${N_LAYER:-12}
export N_HEAD=${N_HEAD:-12}
export N_EMBD=${N_EMBD:-768}
export BATCH_SIZE=${BATCH_SIZE:-1}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
export MAX_STEPS=${MAX_STEPS:-50000}
export WARMUP_STEPS=${WARMUP_STEPS:-1000}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LR=${LR:-0.0002}
export AMP=${AMP:-bf16}

cd "${PROJECT_DIR}"

bash scripts/submit_train_crystal_language_mp_wbm.sbatch
