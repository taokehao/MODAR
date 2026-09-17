#!/bin/bash
#SBATCH --job-name=build_extra
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/build_official_extra_sources_%j.out
#SBATCH --error=logs/build_official_extra_sources_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
OUT_DIR=${OUT_DIR:-data_prep/outputs/official_extra_sources_props}
MAP_SIZE_GB=${MAP_SIZE_GB:-512}
ALEXANDRIA_DATASET=${ALEXANDRIA_DATASET:-pbe}
ALEXANDRIA_PAGE_LIMIT=${ALEXANDRIA_PAGE_LIMIT:-100}
OQMD_PAGE_LIMIT=${OQMD_PAGE_LIMIT:-100}
MAX_RECORDS_PER_SOURCE=${MAX_RECORDS_PER_SOURCE:-}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID:-manual}"
echo "Project:   ${PROJECT_DIR}"
echo "Python:    $(command -v python)"
echo "Raw dir:   ${PROJECT_DIR}/${RAW_DIR}"
echo "Out dir:   ${PROJECT_DIR}/${OUT_DIR}"
echo "Dataset:   Alexandria ${ALEXANDRIA_DATASET} + OQMD"
echo "Time:      $(date)"
echo "============================================================"

echo "Step 1/5: download Alexandria ${ALEXANDRIA_DATASET}"
alex_args=(
  --raw-dir "${RAW_DIR}"
  --download-alexandria
  --alexandria-dataset "${ALEXANDRIA_DATASET}"
  --alexandria-page-limit "${ALEXANDRIA_PAGE_LIMIT}"
)
if [ -n "${ALEXANDRIA_LIMIT:-}" ]; then
  alex_args+=(--alexandria-limit "${ALEXANDRIA_LIMIT}")
fi
python -u data_prep/official_multisource/scripts/01_download_extra_sources.py "${alex_args[@]}"

echo "Step 2/5: download OQMD formationenergy"
oqmd_args=(
  --raw-dir "${RAW_DIR}"
  --download-oqmd
  --oqmd-page-limit "${OQMD_PAGE_LIMIT}"
)
if [ -n "${OQMD_LIMIT:-}" ]; then
  oqmd_args+=(--oqmd-limit "${OQMD_LIMIT}")
fi
python -u data_prep/official_multisource/scripts/01_download_extra_sources.py "${oqmd_args[@]}"

echo "Step 3/5: build Alexandria+OQMD LMDBs"
build_args=(
  --raw-dir "${RAW_DIR}"
  --out-dir "${OUT_DIR}"
  --include-alexandria
  --include-oqmd
  --dedupe
  --map-size-gb "${MAP_SIZE_GB}"
)
if [ -n "${MAX_RECORDS_PER_SOURCE}" ]; then
  build_args+=(--max-records-per-source "${MAX_RECORDS_PER_SOURCE}")
fi
python -u data_prep/official_multisource/scripts/02_build_lmdb_extra_sources.py "${build_args[@]}"

echo "Step 4/5: compute normalization stats"
python -u data_prep/official_multisource/scripts/03_compute_prop_stats.py \
  --out-dir "${OUT_DIR}"

echo "Step 5/5: inspect output LMDB"
python -u data_prep/official_multisource/scripts/04_inspect_lmdb.py \
  "${OUT_DIR}/train.lmdb" --n 2

echo "============================================================"
echo "Done:   $(date)"
echo "Output: ${PROJECT_DIR}/${OUT_DIR}"
echo "============================================================"

