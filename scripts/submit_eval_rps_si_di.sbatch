#!/bin/bash
# d–I, formula + known lattice, N=100 on SI-primitive CNRS 126 / RRUFF 269.
#
#   SET=cnrs_profile sbatch --array=0-3 scripts/submit_eval_rps_si_di.sbatch
#   SET=rruff_peaks  sbatch --array=0-8 scripts/submit_eval_rps_si_di.sbatch
#
#SBATCH --job-name=crylm_rpsSI
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/rps_si_di_%A_%a.out
#SBATCH --error=logs/rps_si_di_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_di}
CHECKPOINT=${CHECKPOINT:-checkpoint_last.pt}
# SET can be passed with --export=SET=...; otherwise infer from --job-name.
if [ -z "${SET:-}" ]; then
  case "${SLURM_JOB_NAME:-}" in
    crylm_siCNR) SET=cnrs_profile ;;
    crylm_siRUF) SET=rruff_peaks ;;
    *) SET="" ;;
  esac
fi
SET=${SET:?set SET=cnrs_profile or rruff_peaks}

case "${SET}" in
  cnrs_profile)
    DATA_DIR=${PROJECT_DIR}/data_prep/outputs/rps_cnrs126_si_profile/crystal_language_v2
    OUT_DIR=${PROJECT_DIR}/crystal_language/eval_results/rps_cnrs126_si_profile_di
    NUM_SHARDS=4
    ;;
  rruff_peaks)
    DATA_DIR=${PROJECT_DIR}/data_prep/outputs/rps_rruff269_si_peaks/crystal_language_v2
    OUT_DIR=${PROJECT_DIR}/crystal_language/eval_results/rps_rruff269_si_peaks_di
    NUM_SHARDS=9
    ;;
  *)
    echo "bad SET=${SET}"; exit 1
    ;;
esac

SPLIT=test
COND=pxrd_formula
SHARD_SIZE=32
NUM_SAMPLES=100
SAMPLES_PER_BATCH=${SAMPLES_PER_BATCH:-20}
TEMPERATURE=0.8
TOP_K_SAMPLING=20
MAX_NEW_TOKENS=1024
TOP_KS=1,10,20,50
RERANK_WORKERS=${RERANK_WORKERS:-4}
AMP=bf16
NOISE_PROFILE=none
NOISE_SEED=20260706

N_RECORDS=$(wc -l < "${DATA_DIR}/${SPLIT}.jsonl")
TID=${SLURM_ARRAY_TASK_ID:-0}
SHARD=$(( TID % NUM_SHARDS ))
START_INDEX=$(( SHARD * SHARD_SIZE ))
if [ "${START_INDEX}" -ge "${N_RECORDS}" ]; then
  echo "SKIP shard ${SHARD}: start ${START_INDEX} >= n ${N_RECORDS}"
  exit 0
fi

STEM="${SPLIT}_${COND}_lat1_full_inject0_cd1_rr1_start${START_INDEX}_n${SHARD_SIZE}_s${NUM_SAMPLES}"
mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "============================================================"
echo "Job:        ${SLURM_ARRAY_JOB_ID:-manual}_${TID}"
echo "Set:        ${SET}"
echo "Data:       ${DATA_DIR}  (n=${N_RECORDS})"
echo "Run:        ${RUN_DIR}"
echo "Out:        ${OUT_DIR}"
echo "Shard:      ${SHARD} start=${START_INDEX}"
echo "Time:       $(date)"
echo "============================================================"

if [ -f "${OUT_DIR}/${STEM}_summary.json" ]; then
  echo "SKIP ${STEM} (summary exists)"
  exit 0
fi

python -u crystal_language/evaluate_generation.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --split "${SPLIT}" \
  --condition-set "${COND}" \
  --use-lattice \
  --target-mode full \
  --constrain formula \
  --rerank --rerank-workers "${RERANK_WORKERS}" \
  --amp "${AMP}" \
  --noise-profile "${NOISE_PROFILE}" \
  --noise-seed "${NOISE_SEED}" \
  --store-structures \
  --start-index "${START_INDEX}" \
  --max-records "${SHARD_SIZE}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-ks "${TOP_KS}" \
  --samples-per-batch "${SAMPLES_PER_BATCH}" \
  --temperature "${TEMPERATURE}" \
  --top-k-sampling "${TOP_K_SAMPLING}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo "DONE ${STEM} $(date)"
