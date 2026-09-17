#!/bin/bash
#SBATCH --job-name=build_unified
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/build_official_unified_%j.out
#SBATCH --error=logs/build_official_unified_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}

# Per-source LMDB outputs.
MP_WBM_DIR=${MP_WBM_DIR:-data_prep/outputs/official_mp_wbm_multimodal}
EXTRA_DIR=${EXTRA_DIR:-data_prep/outputs/official_extra_sources_props}
AFLOW_DIR=${AFLOW_DIR:-data_prep/outputs/official_aflow_props}
UNIFIED_DIR=${UNIFIED_DIR:-data_prep/outputs/official_unified_multimodal}

# AFLOW raw shards (produced by download_aflow_catalog_array.sbatch).
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
AFLOW_RAW_DIR=${AFLOW_RAW_DIR:-${RAW_DIR}}
ALEXS_JSONL=${ALEXS_JSONL:-${RAW_DIR}/alexandria_pbe.multimodal.jsonl}
OQMD_JSONL=${OQMD_JSONL:-${RAW_DIR}/oqmd_formationenergy_merged.multimodal.jsonl}
MAP_SIZE_GB=${MAP_SIZE_GB:-1024}
EXTRA_MAP_SIZE_GB=${EXTRA_MAP_SIZE_GB:-1024}
AFLOW_MAP_SIZE_GB=${AFLOW_MAP_SIZE_GB:-512}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:          ${SLURM_JOB_ID:-manual}"
echo "Project:      ${PROJECT_DIR}"
echo "Python:       $(command -v python)"
echo "MP+WBM dir:   ${MP_WBM_DIR}"
echo "Extra dir:    ${EXTRA_DIR}"
echo "AFLOW dir:    ${AFLOW_DIR}"
echo "Unified dir:  ${UNIFIED_DIR}"
echo "Extra raw:    ${ALEXS_JSONL} + ${OQMD_JSONL}"
echo "AFLOW raw:    ${AFLOW_RAW_DIR}"
echo "Time:         $(date)"
echo "============================================================"

echo "Step 1/5: build Alexandria+OQMD LMDB from enhanced JSONL"
python -u data_prep/official_multisource/scripts/02_build_lmdb_extra_sources.py \
  --raw-dir "${RAW_DIR}" \
  --out-dir "${EXTRA_DIR}" \
  --include-alexandria \
  --include-oqmd \
  --alexandria-jsonl "${ALEXS_JSONL}" \
  --oqmd-jsonl "${OQMD_JSONL}" \
  --dedupe \
  --map-size-gb "${EXTRA_MAP_SIZE_GB}"

echo "Step 2/5: build AFLOW LMDB from enhanced raw shards"
python -u data_prep/official_multisource/scripts/02_build_lmdb_aflow.py \
  --raw-dir "${AFLOW_RAW_DIR}" \
  --out-dir "${AFLOW_DIR}" \
  --dedupe \
  --map-size-gb "${AFLOW_MAP_SIZE_GB}"

echo "Step 3/5: merge MP+WBM + Alex/OQMD + AFLOW into unified LMDB"
merge_args=(--out-dir "${UNIFIED_DIR}" --map-size-gb "${MAP_SIZE_GB}" --dedupe)
for d in "${MP_WBM_DIR}" "${EXTRA_DIR}" "${AFLOW_DIR}"; do
  if [ -d "${d}" ] && [ -f "${d}/train.lmdb" ]; then
    merge_args+=(--source "${d}")
  else
    echo "WARN: skipping ${d} (no train.lmdb yet)"
  fi
done
python -u data_prep/official_multisource/scripts/05_merge_unified_lmdb.py "${merge_args[@]}"

echo "Step 4/5: compute property normalization stats on unified train split"
python -u data_prep/official_multisource/scripts/03_compute_prop_stats.py \
  --out-dir "${UNIFIED_DIR}"

echo "Step 5/5: inspect output LMDB"
echo "Inspect first records of unified train.lmdb"
python -u data_prep/official_multisource/scripts/04_inspect_lmdb.py \
  "${UNIFIED_DIR}/train.lmdb" --n 2

echo "============================================================"
echo "Done:    $(date)"
echo "Output:  ${PROJECT_DIR}/${UNIFIED_DIR}"
echo "============================================================"
