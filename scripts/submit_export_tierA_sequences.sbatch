#!/bin/bash
# Stage B materialization: build the deduped Tier-A training corpus as v2
# sequences, ready for crystal_language/train.py.
#
#   step1  unified LMDB (10.2M) --keep-ids (tight Tier-A, 3.52M) ->
#          crystal-native JSONL  (atom_types / frac_coords / lattice_parameters /
#          conditions{pxrd,neutron,xrf,xps,...})
#   step2  crystal-native JSONL --rep v2 -> sequence JSONL (bin tokens + spectra)
#
# CPU-only, single process (both stages stream).
#
#SBATCH --job-name=tierA_export
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tierA_export_%j.out
#SBATCH --error=logs/tierA_export_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxa_prep}

UNIFIED_DIR=${UNIFIED_DIR:-${PROJECT_DIR}/data_prep/outputs/official_unified_multimodal}
KEEP_IDS=${KEEP_IDS:-${PROJECT_DIR}/data_prep/outputs/metadata_index_unified_tight/tier_a_keep_ids.txt}
CN_DIR=${CN_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_native_unified_tierA}
SEQ_DIR=${SEQ_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}

mkdir -p "${PROJECT_DIR}/logs" "${CN_DIR}" "${SEQ_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== step1: filtered crystal-native export (Tier-A keep list) $(date) ==="
python -u data_prep/official_multisource/scripts/export_crystal_native_jsonl.py \
  --data-dir "${UNIFIED_DIR}" \
  --out-dir "${CN_DIR}" \
  --splits train,valid,test \
  --keep-ids "${KEEP_IDS}"

echo "=== step2: v2 sequence export $(date) ==="
python -u crystal_language/data/export_sequences.py \
  --in-dir "${CN_DIR}" \
  --out-dir "${SEQ_DIR}" \
  --splits train,valid,test \
  --rep v2

echo "=== sequence outputs ==="
ls -lh "${SEQ_DIR}"/*.jsonl
echo "DONE $(date)"
