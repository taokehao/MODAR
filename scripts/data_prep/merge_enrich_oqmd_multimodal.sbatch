#!/bin/bash
#SBATCH --job-name=merge_enrich_oqmd
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/merge_enrich_oqmd_%j.out
#SBATCH --error=logs/merge_enrich_oqmd_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
RAW_DIR=${RAW_DIR:-data_prep/raw/official_extra_sources}
MERGED_JSONL=${MERGED_JSONL:-${RAW_DIR}/oqmd_formationenergy_merged.jsonl}
MULTIMODAL_JSONL=${MULTIMODAL_JSONL:-${RAW_DIR}/oqmd_formationenergy_merged.multimodal.jsonl}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID:-manual}"
echo "Raw dir:   ${PROJECT_DIR}/${RAW_DIR}"
echo "Merged:    ${PROJECT_DIR}/${MERGED_JSONL}"
echo "Enhanced:  ${PROJECT_DIR}/${MULTIMODAL_JSONL}"
echo "Python:    $(command -v python)"
echo "Time:      $(date)"
echo "============================================================"

echo "Step 1/2: merge OQMD main + shard JSONL files"
python -u data_prep/official_multisource/scripts/08_merge_oqmd_shards.py \
  --raw-dir "${RAW_DIR}" \
  --include-main \
  --out "${MERGED_JSONL}"

echo "Step 2/2: enrich merged OQMD with neutron + XPS-valence"
python -u data_prep/official_multisource/scripts/06_enrich_raw_multimodal.py \
  --source oqmd \
  --input-jsonl "${MERGED_JSONL}" \
  --output-jsonl "${MULTIMODAL_JSONL}"

echo "Done: $(date)"
