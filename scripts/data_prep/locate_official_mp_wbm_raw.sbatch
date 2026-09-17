#!/bin/bash
#SBATCH --job-name=loc_mpw_raw
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/locate_official_mp_wbm_raw_%j.out
#SBATCH --error=logs/locate_official_mp_wbm_raw_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_mp_wbm}

mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Project: ${PROJECT_DIR}"
echo "Python:  $(command -v python)"
echo "Raw dir: ${PROJECT_DIR}/${RAW_DIR}"
echo "Time:    $(date)"
echo "============================================================"

# python data_prep/official_multisource/scripts/01_download_mp_wbm.py \
#   --raw-dir "${RAW_DIR}" \
#   --locate-wbm

export MP_API_KEY=28R6mvyh4R8tdMi8qhPNxA4dnJDTIn0H
python data_prep/official_multisource/scripts/01_download_mp_wbm.py \
  --raw-dir data_prep/raw/official_mp_wbm \
  --download-mp

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"

