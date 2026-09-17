#!/bin/bash
# P2/P3 (tight): metadata + cell-metric near-duplicate index + Tier-A tagging
# over the UNIFIED ~10.2M multimodal corpus.
#
# Tight dedup key = (reduced_formula, space_group, n_sites,
#   sorted cell lengths @ 0.1 A grid, sorted angles @ 1 deg grid) + exact hash.
# Keeps distinct polymorphs apart (vs the coarse round(vpa) key) while still
# merging the same structure relaxed across sources. Source priority for the
# kept representative: MP > WBM > OQMD > Alexandria > AFLOW.
#
#SBATCH --job-name=meta_uni_tight
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/metadata_index_unified_tight_%j.out
#SBATCH --error=logs/metadata_index_unified_tight_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxa_prep}
UNIFIED_DIR=${UNIFIED_DIR:-${PROJECT_DIR}/data_prep/outputs/official_unified_multimodal}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/data_prep/outputs/metadata_index_unified_tight}
LEN_GRID=${LEN_GRID:-0.1}
ANGLE_GRID=${ANGLE_GRID:-1.0}

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Unified TIGHT metadata index: ${UNIFIED_DIR} -> ${OUT_DIR}"
echo "  len-grid=${LEN_GRID} A  angle-grid=${ANGLE_GRID} deg  ($(date))"

python -u data_prep/official_multisource/scripts/build_metadata_index.py \
  --jsonl "${UNIFIED_DIR}/train.lmdb" \
  --jsonl "${UNIFIED_DIR}/valid.lmdb" \
  --jsonl "${UNIFIED_DIR}/test.lmdb" \
  --out-dir "${OUT_DIR}" \
  --dedup-mode tight \
  --len-grid "${LEN_GRID}" \
  --angle-grid "${ANGLE_GRID}"

echo "Done: $(date)"
