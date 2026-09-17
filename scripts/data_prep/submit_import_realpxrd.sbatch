#!/bin/bash
#SBATCH --job-name=rpxrd_import
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/import_realpxrd_%j.out
#SBATCH --error=logs/import_realpxrd_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxa_prep}
WORKERS=${WORKERS:-64}
LMDB_DIR=${LMDB_DIR:-/home/tkh/python-project/alex_aflow_oqmd_mp/datasets}
NATIVE_DIR=${NATIVE_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_native_realpxrd}
SEQ_DIR=${SEQ_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_realpxrd_v2}

cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID:-manual}   workers=${WORKERS}"
echo "LMDB dir:  ${LMDB_DIR}"
echo "Native:    ${NATIVE_DIR}"
echo "Seq:       ${SEQ_DIR}"
echo "Time:      $(date)"
echo "============================================================"

IMPORT=data_prep/official_multisource/scripts/import_realpxrd.py

# Small splits first (cheap; also validate the pipeline before the 6M run).
srun python -u "${IMPORT}" --split valid --lmdb-dir "${LMDB_DIR}" --out-dir "${NATIVE_DIR}" --workers "${WORKERS}" --canonical
srun python -u "${IMPORT}" --split test  --lmdb-dir "${LMDB_DIR}" --out-dir "${NATIVE_DIR}" --workers "${WORKERS}" --canonical
# Full 6.088M train split.
srun python -u "${IMPORT}" --split train --lmdb-dir "${LMDB_DIR}" --out-dir "${NATIVE_DIR}" --workers "${WORKERS}" --canonical

# V2 bin-token sequence export for the language-model route.
srun python -u crystal_language/data/export_sequences.py \
  --in-dir "${NATIVE_DIR}" \
  --out-dir "${SEQ_DIR}" \
  --rep v2

echo "DONE $(date)"
