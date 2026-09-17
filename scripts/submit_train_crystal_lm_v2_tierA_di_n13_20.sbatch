#!/bin/bash
#SBATCH --job-name=crylm_n1320
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4090D:4
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/crystal_lm_v2_tierA_di_n13_20_%j.out
#SBATCH --error=logs/crystal_lm_v2_tierA_di_n13_20_%j.err

# 13–20 site specialist for the RPS SI band. Does NOT replace the paper mix
# model (tierA_s2_di) or the full-corpus RPS-lat head (tierA_s2_di_rps_formula_lat).
#
# Inference routing vs RealPXRD-Solver:
#   n_sites < 13  -> mix d–I (tierA_s2_di), MACE energy @1 / CHGNet fusion @20
#   n_sites 13–20 -> this specialist, same decode (T=1.0 / k=50, N=100)
#
# Recipe: copy the validated bigcell specialist, but window 13–20 instead of
# min_sites=21, warm-start the d–I mix checkpoint, and lock the RPS protocol
# (PXRD + formula + known lattice). Data is the Tier-A 13–20 slice (~560k).
# On this shared node prefer scripts/run_train_n13_20_leftover.py so we never
# steal VASP/ColabFold GPUs.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-4}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_di_n13_20}
BASE_RUN=${BASE_RUN:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_di}
INIT_FROM=${INIT_FROM:-${BASE_RUN}/checkpoint_last.pt}

MIN_SITES=${MIN_SITES:-13}
MAX_SITES=${MAX_SITES:-20}
CONDITION_SET=${CONDITION_SET:-pxrd_formula}
USE_LATTICE=${USE_LATTICE:-always}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
BATCH_SIZE=${BATCH_SIZE:-12}
GRAD_ACCUM=${GRAD_ACCUM:-4}
# Size-window specialist, same LR as historical bigcell (min_sites=21).
LR=${LR:-1e-4}
# ~6 epochs over ~560k at effective batch 192: 6*560000/192 ~= 17500.
MAX_STEPS=${MAX_STEPS:-18000}
WARMUP_STEPS=${WARMUP_STEPS:-500}
EVAL_EVERY=${EVAL_EVERY:-1000}
SAVE_EVERY=${SAVE_EVERY:-2000}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}
NOISE_PROFILE=${NOISE_PROFILE:-experimental}
SPECTRA_MAX_PEAKS=${SPECTRA_MAX_PEAKS:-128}
SPECTRA_ENCODER_LAYERS=${SPECTRA_ENCODER_LAYERS:-3}

mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

for split in train valid test; do
  if [[ ! -s "${DATA_DIR}/${split}.jsonl" ]]; then
    echo "FATAL: missing ${DATA_DIR}/${split}.jsonl" >&2
    exit 1
  fi
done
if [[ ! -s "${INIT_FROM}" ]]; then
  echo "FATAL: init checkpoint missing: ${INIT_FROM}" >&2
  exit 1
fi
if [[ ! -s "${RUN_DIR}/tokenizer.json" ]]; then
  cp "${BASE_RUN}/tokenizer.json" "${RUN_DIR}/tokenizer.json"
  echo "copied tokenizer.json from ${BASE_RUN} into ${RUN_DIR}"
fi
ls -lh "${DATA_DIR}"/*.jsonl

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} run_dir=${RUN_DIR} init=${INIT_FROM}"
echo "window n_sites=[${MIN_SITES}, ${MAX_SITES}] condition=${CONDITION_SET} use_lattice=${USE_LATTICE}"
echo "paper mix run untouched: ${BASE_RUN}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

COMMON_ARGS=(
  --arch v2 --rep v2
  --data-dir "${DATA_DIR}"
  --out-dir "${RUN_DIR}"
  --condition-set "${CONDITION_SET}"
  --use-lattice "${USE_LATTICE}"
  --target-mode "${TARGET_MODE}"
  --block-size "${BLOCK_SIZE}"
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}"
  --spectra-encoder di
  --spectra-max-peaks "${SPECTRA_MAX_PEAKS}"
  --spectra-encoder-layers "${SPECTRA_ENCODER_LAYERS}"
  --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}"
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}"
  --eval-every "${EVAL_EVERY}" --save-every "${SAVE_EVERY}"
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
  --cond-adanorm --cond-drop "${COND_DROP}"
  --coord-value-emb
  --noise-profile "${NOISE_PROFILE}"
  --lazy-data auto
  --init-from "${INIT_FROM}"
  --min-sites "${MIN_SITES}"
  --max-sites "${MAX_SITES}"
)

echo "=== prepare: tokenizer + lazy index caches $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
