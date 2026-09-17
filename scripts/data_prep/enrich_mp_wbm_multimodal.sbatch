#!/bin/bash
#SBATCH --job-name=enrich_mpw_mm
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/enrich_mp_wbm_multimodal_%j.out
#SBATCH --error=logs/enrich_mp_wbm_multimodal_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
IN_DIR=${IN_DIR:-data_prep/outputs/official_mp_wbm_props}
OUT_DIR=${OUT_DIR:-data_prep/outputs/official_mp_wbm_multimodal}
MAP_SIZE_GB=${MAP_SIZE_GB:-256}
SPLITS=${SPLITS:-train,valid,test}
MAX_RECORDS_PER_SPLIT=${MAX_RECORDS_PER_SPLIT:-}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:     ${SLURM_JOB_ID:-manual}"
echo "Input:   ${PROJECT_DIR}/${IN_DIR}"
echo "Output:  ${PROJECT_DIR}/${OUT_DIR}"
echo "Splits:  ${SPLITS}"
echo "Python:  $(command -v python)"
echo "Time:    $(date)"
echo "============================================================"

args=(
  --in-dir "${IN_DIR}"
  --out-dir "${OUT_DIR}"
  --splits "${SPLITS}"
  --map-size-gb "${MAP_SIZE_GB}"
)
if [ -n "${MAX_RECORDS_PER_SPLIT}" ]; then
  args+=(--max-records-per-split "${MAX_RECORDS_PER_SPLIT}")
fi

python -u data_prep/official_multisource/scripts/07_enrich_lmdb_multimodal.py "${args[@]}"

echo "============================================================"
echo "Done:   $(date)"
echo "Output: ${PROJECT_DIR}/${OUT_DIR}"
echo "============================================================"
