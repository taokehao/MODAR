#!/bin/bash
# RealPXRD-Solver all-L on SI-preprocessed CNRS 126 / RRUFF 269.
#
# Array:
#   0 cnrs126  Top-1
#   1 cnrs126  Top-20
#   2 rruff269 Top-1
#   3 rruff269 Top-20
#
#SBATCH --job-name=rps_si
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/rps_si_%A_%a.out
#SBATCH --error=logs/rps_si_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
VENV=${VENV:-${PROJECT_DIR}/.venv_rps}
MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/data_prep/realpxrd_solver/weight/2501/pxrd-all-L}
LMDB_DIR=${LMDB_DIR:-${PROJECT_DIR}/data_prep/realpxrd_solver/lmdb_si}
OUT_ROOT=${OUT_ROOT:-${PROJECT_DIR}/crystal_language/eval_results/realpxrd_solver_si}

TID=${SLURM_ARRAY_TASK_ID:-0}
case "${TID}" in
  0) SET=cnrs126  ; NEV=1  ;;
  1) SET=cnrs126  ; NEV=20 ;;
  2) SET=rruff269 ; NEV=1  ;;
  3) SET=rruff269 ; NEV=20 ;;
  *) echo "bad task id ${TID}"; exit 1 ;;
esac

TEST_DATA=${LMDB_DIR}/${SET}.lmdb
OUT_DIR=${OUT_ROOT}/${SET}_allL
mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/logs"

echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} set=${SET} num_evals=${NEV}"
echo "model=${MODEL_PATH}"
echo "data=${TEST_DATA}"

source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

cd "${PROJECT_DIR}"
python scripts/run_realpxrd_solver_eval.py \
  --model_path "${MODEL_PATH}" \
  --test_data "${TEST_DATA}" \
  --out_dir "${OUT_DIR}" \
  --num_evals "${NEV}" \
  --batch_size "${BATCH_SIZE:-16}"
