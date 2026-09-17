#!/bin/bash
#SBATCH --job-name=dl_oqmd
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/download_oqmd_%j.out
#SBATCH --error=logs/download_oqmd_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
OQMD_PAGE_LIMIT=${OQMD_PAGE_LIMIT:-100}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:            ${SLURM_JOB_ID:-manual}"
echo "Project:        ${PROJECT_DIR}"
echo "Python:         $(command -v python)"
echo "Raw dir:        ${PROJECT_DIR}/${RAW_DIR}"
echo "OQMD page size: ${OQMD_PAGE_LIMIT}"
echo "Time:           $(date)"
echo "============================================================"

args=(
  --raw-dir "${RAW_DIR}"
  --download-oqmd
  --oqmd-page-limit "${OQMD_PAGE_LIMIT}"
)
if [ -n "${OQMD_LIMIT:-}" ]; then
  args+=(--oqmd-limit "${OQMD_LIMIT}")
fi

python -u data_prep/official_multisource/scripts/01_download_extra_sources.py "${args[@]}"

echo "============================================================"
echo "Done: $(date)"
echo "Output: ${PROJECT_DIR}/${RAW_DIR}/oqmd_formationenergy.jsonl"
echo "============================================================"
