#!/bin/bash
#SBATCH --job-name=dl_aflow
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/download_aflow_%x_%A_%a.out
#SBATCH --error=logs/download_aflow_%x_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
AFLOW_CATALOG=${AFLOW_CATALOG:-LIB2_RAW}
AFLOW_SHARD_COUNT=${AFLOW_SHARD_COUNT:-4}
AFLOW_SYSTEM_LIST=${AFLOW_SYSTEM_LIST:-}
AFLOW_TIMEOUT_S=${AFLOW_TIMEOUT_S:-90}
AFLOW_SLEEP_S=${AFLOW_SLEEP_S:-0.2}
AFLOW_WORKERS=${AFLOW_WORKERS:-16}
AFLOW_LIMIT=${AFLOW_LIMIT:-}
HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:38080}
HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:38080}
ALL_PROXY=${ALL_PROXY:-socks5://127.0.0.1:38081}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HTTP_PROXY HTTPS_PROXY ALL_PROXY
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export all_proxy="${ALL_PROXY}"

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================================"
echo "Job:      ${SLURM_JOB_ID:-manual}.${SHARD_INDEX}"
echo "Catalog:  ${AFLOW_CATALOG}"
echo "Shard:    ${SHARD_INDEX}/${AFLOW_SHARD_COUNT}"
echo "Systems:  ${AFLOW_SYSTEM_LIST:-auto}"
echo "Raw dir:  ${PROJECT_DIR}/${RAW_DIR}"
echo "Python:   $(command -v python)"
echo "Proxy:    ${HTTPS_PROXY}"
echo "Workers:  ${AFLOW_WORKERS}"
echo "Time:     $(date)"
echo "============================================================"

args=(
  --raw-dir "${RAW_DIR}"
  --download-aflow-catalog
  --aflow-catalog "${AFLOW_CATALOG}"
  --aflow-shard-index "${SHARD_INDEX}"
  --aflow-shard-count "${AFLOW_SHARD_COUNT}"
  --aflow-timeout-s "${AFLOW_TIMEOUT_S}"
  --aflow-sleep-s "${AFLOW_SLEEP_S}"
  --aflow-workers "${AFLOW_WORKERS}"
)

if [ -n "${AFLOW_SYSTEM_LIST}" ]; then
  args+=(--aflow-system-list "${AFLOW_SYSTEM_LIST}")
fi

if [ -n "${AFLOW_LIMIT}" ]; then
  args+=(--aflow-limit "${AFLOW_LIMIT}")
fi

python -u data_prep/official_multisource/scripts/01_download_extra_sources.py "${args[@]}"

echo "Done: $(date)"

