#!/bin/bash
#SBATCH --job-name=merge_unified
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/merge_unified_parts_%j.out
#SBATCH --error=logs/merge_unified_parts_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
MP_WBM_DIR=${MP_WBM_DIR:-data_prep/outputs/official_mp_wbm_multimodal}
EXTRA_PARTS_ROOT=${EXTRA_PARTS_ROOT:-data_prep/outputs/official_extra_sources_parts}
AFLOW_PARTS_ROOT=${AFLOW_PARTS_ROOT:-data_prep/outputs/official_aflow_parts}
UNIFIED_DIR=${UNIFIED_DIR:-data_prep/outputs/official_unified_multimodal}
MAP_SIZE_GB=${MAP_SIZE_GB:-1024}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:          ${SLURM_JOB_ID:-manual}"
echo "MP+WBM:       ${MP_WBM_DIR}"
echo "Extra parts:  ${EXTRA_PARTS_ROOT}"
echo "AFLOW parts:  ${AFLOW_PARTS_ROOT}"
echo "Unified dir:  ${UNIFIED_DIR}"
echo "Python:       $(command -v python)"
echo "Time:         $(date)"
echo "============================================================"

merge_args=(--out-dir "${UNIFIED_DIR}" --map-size-gb "${MAP_SIZE_GB}" --dedupe)
if [ -f "${MP_WBM_DIR}/train.lmdb" ]; then
  merge_args+=(--source "${MP_WBM_DIR}")
fi
for d in "${EXTRA_PARTS_ROOT}"/part_* "${AFLOW_PARTS_ROOT}"/part_*; do
  if [ -f "${d}/train.lmdb" ]; then
    merge_args+=(--source "${d}")
  else
    echo "WARN: missing train.lmdb in ${d}; skipping"
  fi
done

echo "Merging ${#merge_args[@]} arguments"
python -u data_prep/official_multisource/scripts/05_merge_unified_lmdb.py "${merge_args[@]}"

echo "Compute normalization stats"
python -u data_prep/official_multisource/scripts/03_compute_prop_stats.py \
  --out-dir "${UNIFIED_DIR}"

echo "Inspect output"
python -u data_prep/official_multisource/scripts/04_inspect_lmdb.py \
  "${UNIFIED_DIR}/train.lmdb" --n 2

echo "Done: $(date)"
