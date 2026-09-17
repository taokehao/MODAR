#!/bin/bash
#SBATCH --job-name=build_extra_arr
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/build_extra_lmdb_array_%A_%a.out
#SBATCH --error=logs/build_extra_lmdb_array_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
OUT_ROOT=${OUT_ROOT:-data_prep/outputs/official_extra_sources_parts}
SHARD_COUNT=${SHARD_COUNT:-16}
MAP_SIZE_GB=${MAP_SIZE_GB:-128}
ALEXS_JSONL=${ALEXS_JSONL:-${RAW_DIR}/alexandria_pbe.multimodal.jsonl}
OQMD_JSONL=${OQMD_JSONL:-${RAW_DIR}/oqmd_formationenergy_merged.multimodal.jsonl}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
OUT_DIR="${OUT_ROOT}/part_${SHARD_INDEX}_of_${SHARD_COUNT}"

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID:-manual}.${SHARD_INDEX}"
echo "Shard:     ${SHARD_INDEX}/${SHARD_COUNT}"
echo "Out dir:   ${PROJECT_DIR}/${OUT_DIR}"
echo "Raw:       ${ALEXS_JSONL} + ${OQMD_JSONL}"
echo "Python:    $(command -v python)"
echo "Time:      $(date)"
echo "============================================================"

python -u data_prep/official_multisource/scripts/02_build_lmdb_extra_sources.py \
  --raw-dir "${RAW_DIR}" \
  --out-dir "${OUT_DIR}" \
  --include-alexandria \
  --include-oqmd \
  --alexandria-jsonl "${ALEXS_JSONL}" \
  --oqmd-jsonl "${OQMD_JSONL}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SHARD_COUNT}" \
  --map-size-gb "${MAP_SIZE_GB}"

echo "Done: $(date)"
