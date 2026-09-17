#!/bin/bash
# P2/P3: metadata + coarse near-duplicate index + Tier-A tagging over the
# UNIFIED ~10.2M multimodal corpus (the database that feeds Tier-A pretraining).
# CPU-only. Streams the compressed LMDB (train 44 GB + valid + test) once.
# Coarse near-dup dedup keeps one representative per group by source priority
# MP > WBM > OQMD > Alexandria > AFLOW (see build_metadata_index.py).
#
#SBATCH --job-name=meta_unified
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/metadata_index_unified_%j.out
#SBATCH --error=logs/metadata_index_unified_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxa_prep}
UNIFIED_DIR=${UNIFIED_DIR:-${PROJECT_DIR}/data_prep/outputs/official_unified_multimodal}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/data_prep/outputs/metadata_index_unified}

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Unified metadata index: ${UNIFIED_DIR} -> ${OUT_DIR}  ($(date))"

python -u data_prep/official_multisource/scripts/build_metadata_index.py \
  --jsonl "${UNIFIED_DIR}/train.lmdb" \
  --jsonl "${UNIFIED_DIR}/valid.lmdb" \
  --jsonl "${UNIFIED_DIR}/test.lmdb" \
  --out-dir "${OUT_DIR}"

echo "Done: $(date)"
