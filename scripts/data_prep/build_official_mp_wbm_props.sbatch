#!/bin/bash
#SBATCH --job-name=build_mpw
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/build_official_mp_wbm_props_%j.out
#SBATCH --error=logs/build_official_mp_wbm_props_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_mp_wbm}
OUT_DIR=${OUT_DIR:-data_prep/outputs/official_mp_wbm_props}
MAP_SIZE_GB=${MAP_SIZE_GB:-128}
MP_CHUNK_SIZE=${MP_CHUNK_SIZE:-1000}
MAX_RECORDS_PER_SOURCE=${MAX_RECORDS_PER_SOURCE:-}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:     ${SLURM_JOB_ID:-manual}"
echo "Project: ${PROJECT_DIR}"
echo "Python:  $(command -v python)"
echo "Raw dir: ${PROJECT_DIR}/${RAW_DIR}"
echo "Out dir: ${PROJECT_DIR}/${OUT_DIR}"
echo "Time:    $(date)"
echo "============================================================"

if [ -z "${MP_API_KEY:-}" ] && [ -z "${MAPI_KEY:-}" ]; then
  echo "ERROR: MP_API_KEY or MAPI_KEY is required for MP download." >&2
  exit 2
fi

echo "Step 1/5: locate WBM raw files"
python data_prep/official_multisource/scripts/01_download_mp_wbm.py \
  --raw-dir "${RAW_DIR}" \
  --locate-wbm

echo "Step 2/5: download MP official summary docs"
mp_args=(--raw-dir "${RAW_DIR}" --download-mp --mp-chunk-size "${MP_CHUNK_SIZE}")
if [ -n "${MP_LIMIT:-}" ]; then
  mp_args+=(--mp-limit "${MP_LIMIT}")
fi
python data_prep/official_multisource/scripts/01_download_mp_wbm.py "${mp_args[@]}"

echo "Step 3/5: build MP+WBM LMDBs"
build_args=(
  --raw-dir "${RAW_DIR}"
  --out-dir "${OUT_DIR}"
  --include-mp
  --include-wbm
  --dedupe
  --map-size-gb "${MAP_SIZE_GB}"
)
if [ -n "${MAX_RECORDS_PER_SOURCE}" ]; then
  build_args+=(--max-records-per-source "${MAX_RECORDS_PER_SOURCE}")
fi
python data_prep/official_multisource/scripts/02_build_lmdb_mp_wbm.py "${build_args[@]}"

echo "Step 4/5: compute normalization stats"
python data_prep/official_multisource/scripts/03_compute_prop_stats.py \
  --out-dir "${OUT_DIR}"

echo "Step 5/5: inspect output LMDB"
python data_prep/official_multisource/scripts/04_inspect_lmdb.py \
  "${OUT_DIR}/train.lmdb" --n 2

echo "============================================================"
echo "Done: $(date)"
echo "Output: ${PROJECT_DIR}/${OUT_DIR}"
echo "============================================================"
