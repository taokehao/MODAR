#!/bin/bash
#SBATCH --job-name=dl_oqmd_shard
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/download_oqmd_shard_%A_%a.out
#SBATCH --error=logs/download_oqmd_shard_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
OQMD_PAGE_LIMIT=${OQMD_PAGE_LIMIT:-100}
OQMD_SHARD_COUNT=${OQMD_SHARD_COUNT:-16}
OQMD_TOTAL_RECORDS=${OQMD_TOTAL_RECORDS:-1407395}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================================"
echo "Job:          ${SLURM_JOB_ID:-manual}.${SHARD_INDEX}"
echo "Project:      ${PROJECT_DIR}"
echo "Raw dir:      ${PROJECT_DIR}/${RAW_DIR}"
echo "Shard:        ${SHARD_INDEX}/${OQMD_SHARD_COUNT}"
echo "Total:        ${OQMD_TOTAL_RECORDS}"
echo "Page size:    ${OQMD_PAGE_LIMIT}"
echo "Python:       $(command -v python)"
echo "Time:         $(date)"
echo "============================================================"

python -u data_prep/official_multisource/scripts/01_download_extra_sources.py \
  --raw-dir "${RAW_DIR}" \
  --download-oqmd-shard \
  --oqmd-shard-index "${SHARD_INDEX}" \
  --oqmd-shard-count "${OQMD_SHARD_COUNT}" \
  --oqmd-total-records "${OQMD_TOTAL_RECORDS}" \
  --oqmd-page-limit "${OQMD_PAGE_LIMIT}"

echo "Done: $(date)"
