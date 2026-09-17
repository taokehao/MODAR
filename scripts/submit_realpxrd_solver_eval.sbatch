#!/bin/bash
# Run RealPXRD-Solver (authors' ckpt) on their CNRS/RRUFF and our CNRS 205.
#
# Array:
#   0 cnrs126  Top-1
#   1 cnrs126  Top-20
#   2 cnrs205  Top-1
#   3 cnrs205  Top-20
#   4 rruff269 Top-1
#   5 rruff269 Top-20
#
# Smoke (one batch, Top-1):
#   ONEBATCH=1 sbatch --array=0%1 scripts/submit_realpxrd_solver_eval.sbatch
#
#SBATCH --job-name=rps_eval
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/rps_eval_%A_%a.out
#SBATCH --error=logs/rps_eval_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
VENV=${VENV:-${PROJECT_DIR}/.venv_rps}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/data_prep/realpxrd_solver/weight/2501/pxrd-atom25-L}
LMDB_DIR=${LMDB_DIR:-${PROJECT_DIR}/data_prep/realpxrd_solver/lmdb}
OUT_ROOT=${OUT_ROOT:-${PROJECT_DIR}/crystal_language/eval_results/realpxrd_solver}
ONEBATCH=${ONEBATCH:-0}

TID=${SLURM_ARRAY_TASK_ID:-0}
case "${TID}" in
  0) SET=cnrs126  ; NEV=1  ;;
  1) SET=cnrs126  ; NEV=20 ;;
  2) SET=cnrs205  ; NEV=1  ;;
  3) SET=cnrs205  ; NEV=20 ;;
  4) SET=rruff269 ; NEV=1  ;;
  5) SET=rruff269 ; NEV=20 ;;
  *) echo "bad task id ${TID}"; exit 1 ;;
esac

TEST_DATA=${LMDB_DIR}/${SET}.lmdb
OUT_DIR=${OUT_ROOT}/${SET}_atom25L
mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/logs"

echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} set=${SET} num_evals=${NEV} onebatch=${ONEBATCH}"
echo "model=${MODEL_PATH}"
echo "data=${TEST_DATA}"

source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

EXTRA=()
if [ "${ONEBATCH}" = "1" ]; then
  EXTRA+=(--onebatch)
fi

cd "${PROJECT_DIR}"
python scripts/run_realpxrd_solver_eval.py \
  --model_path "${MODEL_PATH}" \
  --test_data "${TEST_DATA}" \
  --out_dir "${OUT_DIR}" \
  --num_evals "${NEV}" \
  --batch_size "${BATCH_SIZE:-16}" \
  "${EXTRA[@]}"
