#!/bin/bash
#SBATCH --job-name=build_aflow_arr
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/build_aflow_lmdb_array_%A_%a.out
#SBATCH --error=logs/build_aflow_lmdb_array_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
OUT_ROOT=${OUT_ROOT:-data_prep/outputs/official_aflow_parts}
MAP_SIZE_GB=${MAP_SIZE_GB:-128}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t AFLOW_FILES < <(compgen -G "${RAW_DIR}/aflow_lib*_raw_shard_*_of_004.multimodal.jsonl" | sort)
if [ "${#AFLOW_FILES[@]}" -eq 0 ]; then
  echo "ERROR: no enhanced AFLOW shard JSONL found in ${RAW_DIR}" >&2
  exit 2
fi
if [ "${SHARD_INDEX}" -ge "${#AFLOW_FILES[@]}" ]; then
  echo "ERROR: shard index ${SHARD_INDEX} out of range for ${#AFLOW_FILES[@]} files" >&2
  exit 2
fi
INPUT_JSONL=${AFLOW_FILES[$SHARD_INDEX]}
OUT_DIR="${OUT_ROOT}/part_${SHARD_INDEX}_of_${#AFLOW_FILES[@]}"

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID:-manual}.${SHARD_INDEX}"
echo "Input:     ${INPUT_JSONL}"
echo "Out dir:   ${PROJECT_DIR}/${OUT_DIR}"
echo "Python:    $(command -v python)"
echo "Time:      $(date)"
echo "============================================================"

python -u data_prep/official_multisource/scripts/02_build_lmdb_aflow.py \
  --raw-dir "${RAW_DIR}" \
  --out-dir "${OUT_DIR}" \
  --shard "${INPUT_JSONL}" \
  --map-size-gb "${MAP_SIZE_GB}"

echo "Done: $(date)"
