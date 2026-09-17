#!/bin/bash
#SBATCH --job-name=leak_audit
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/leakage_audit_%j.out
#SBATCH --error=logs/leakage_audit_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

TRAIN=${TRAIN:-${PROJECT_DIR}/data_prep/outputs/crystal_native_mp_wbm/train.jsonl}
EVAL1=${EVAL1:-${PROJECT_DIR}/data_prep/outputs/crystal_native_mp_wbm/valid.jsonl}
EVAL2=${EVAL2:-${PROJECT_DIR}/data_prep/outputs/crystal_native_mp_wbm/test.jsonl}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/data_prep/outputs/leakage_audit_mp_wbm}
WORKERS=${WORKERS:-16}

cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

srun python -u data_prep/official_multisource/scripts/leakage_audit.py \
  --train "${TRAIN}" \
  --eval "${EVAL1}" \
  --eval "${EVAL2}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}"
