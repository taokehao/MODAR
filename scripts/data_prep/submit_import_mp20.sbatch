#!/bin/bash
#SBATCH --job-name=mp20_import
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/import_mp20_%j.out
#SBATCH --error=logs/import_mp20_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}
WORKERS=${WORKERS:-32}

cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

srun python -u data_prep/official_multisource/scripts/import_mp20.py --workers "${WORKERS}"

# Sequence export for the language-model route (fast, single process).
srun python -u crystal_language/data/export_sequences.py \
  --in-dir data_prep/outputs/crystal_native_mp20 \
  --out-dir data_prep/outputs/crystal_language_mp20
