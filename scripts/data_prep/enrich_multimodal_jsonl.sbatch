#!/bin/bash
#SBATCH --job-name=enrich_mm
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/enrich_multimodal_%A_%a.out
#SBATCH --error=logs/enrich_multimodal_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-uni3dar}
SOURCE=${SOURCE:?Set SOURCE=alexandria, oqmd, or aflow}
INPUT_JSONL=${INPUT_JSONL:-}
INPUT_GLOB=${INPUT_GLOB:-}
OUTPUT_JSONL=${OUTPUT_JSONL:-}
MAX_RECORDS=${MAX_RECORDS:-}

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"

if [ -z "${INPUT_JSONL}" ] && [ -n "${INPUT_GLOB}" ]; then
  mapfile -t INPUT_FILES < <(compgen -G "${INPUT_GLOB}" | sort)
  if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "ERROR: INPUT_GLOB matched no files: ${INPUT_GLOB}" >&2
    exit 2
  fi
  IDX=${SLURM_ARRAY_TASK_ID:-0}
  if [ "${IDX}" -ge "${#INPUT_FILES[@]}" ]; then
    echo "ERROR: array index ${IDX} out of range for ${#INPUT_FILES[@]} files" >&2
    exit 2
  fi
  INPUT_JSONL=${INPUT_FILES[$IDX]}
fi
if [ -z "${INPUT_JSONL}" ]; then
  echo "ERROR: set INPUT_JSONL or INPUT_GLOB" >&2
  exit 2
fi

echo "============================================================"
echo "Job:     ${SLURM_JOB_ID:-manual}.${SLURM_ARRAY_TASK_ID:-0}"
echo "Source:  ${SOURCE}"
echo "Input:   ${INPUT_JSONL}"
echo "Output:  ${OUTPUT_JSONL:-auto}"
echo "Python:  $(command -v python)"
echo "Time:    $(date)"
echo "============================================================"

args=(
  --source "${SOURCE}"
  --input-jsonl "${INPUT_JSONL}"
)
if [ -n "${OUTPUT_JSONL}" ]; then
  args+=(--output-jsonl "${OUTPUT_JSONL}")
fi
if [ -n "${MAX_RECORDS}" ]; then
  args+=(--max-records "${MAX_RECORDS}")
fi

python -u data_prep/official_multisource/scripts/06_enrich_raw_multimodal.py "${args[@]}"

echo "Done: $(date)"
