#!/bin/bash
# Score a finished N=100 known-lattice eval pool with CHGNet.
#
#   EVAL_DIR=.../eval_results/rps_cnrs126_si_profile_di \
#     OUT=.../eval_results/chgnet/rps_cnrs126_si_profile_di.jsonl \
#     sbatch --dependency=afterok:<eval_jobid> scripts/submit_score_chgnet.sbatch
#
#SBATCH --job-name=crylm_chgnet
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chgnet_%j.out
#SBATCH --error=logs/chgnet_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

if [ -z "${POOL:-}" ]; then
  case "${SLURM_JOB_NAME:-}" in
    crylm_chgCNR) POOL=rps_cnrs126_si_profile_di ;;
    crylm_chgRUF) POOL=rps_rruff269_si_peaks_di ;;
    *) POOL="" ;;
  esac
fi
if [ -n "${POOL}" ]; then
  EVAL_DIR=${PROJECT_DIR}/crystal_language/eval_results/${POOL}
  OUT=${PROJECT_DIR}/crystal_language/eval_results/chgnet/${POOL}.jsonl
else
  EVAL_DIR=${EVAL_DIR:?set EVAL_DIR or POOL}
  OUT=${OUT:?set OUT or POOL}
fi
COND=${COND:-pxrd_formula}

mkdir -p "${PROJECT_DIR}/logs" "$(dirname "${OUT}")"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "CHGNet score  eval=${EVAL_DIR}  out=${OUT}  $(date)"
python -u scripts/score_candidates_chgnet.py \
  --eval-dir "${EVAL_DIR}" \
  --out "${OUT}" \
  --cond "${COND}"
echo "DONE $(date)"
