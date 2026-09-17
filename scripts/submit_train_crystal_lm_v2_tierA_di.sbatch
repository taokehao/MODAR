#!/bin/bash
#SBATCH --job-name=crylm_tierA_di
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4090D:4
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/crystal_lm_v2_tierA_di_%j.out
#SBATCH --error=logs/crystal_lm_v2_tierA_di_%j.err

# Stage B variant: swap the PXRD/ND front-end from a fixed 2-theta raster CNN to
# a (d, intensity) peak-list transformer, so the conditioning no longer depends
# on wavelength, angular range, or step size. The decoder is warm-started from
# the raster Stage-B run and only the spectra encoder is re-initialized.
#
# The Tier-A sequence JSONL already stores discrete peak lists, so no data
# re-export is needed: the dataset converts 2-theta to d at load time.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-4}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_di}
INIT_FROM=${INIT_FROM:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve/checkpoint_last.pt}

CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
BATCH_SIZE=${BATCH_SIZE:-12}
GRAD_ACCUM=${GRAD_ACCUM:-4}
# Warm-started decoder plus a fresh encoder: a lower peak LR than the 3e-4 used
# from scratch keeps the pretrained decoder from being washed out while the new
# encoder catches up.
LR=${LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-20000}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}
# Peak-list augmentation covers weak-peak dropping, near-peak merging, position
# jitter, spurious peaks, angular-range truncation and step quantization.
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
    echo "FATAL: missing ${DATA_DIR}/${split}.jsonl (seq export incomplete?)" >&2
    exit 1
  fi
done
if [[ ! -s "${INIT_FROM}" ]]; then
  echo "FATAL: init checkpoint missing: ${INIT_FROM}" >&2
  exit 1
fi
# The vocabulary is unchanged by the front-end swap, so reuse the source
# tokenizer rather than rebuilding it from the corpus.
if [[ ! -s "${RUN_DIR}/tokenizer.json" ]]; then
  cp "$(dirname "${INIT_FROM}")/tokenizer.json" "${RUN_DIR}/tokenizer.json"
  echo "copied tokenizer.json from $(dirname "${INIT_FROM}") into ${RUN_DIR}"
fi
ls -lh "${DATA_DIR}"/*.jsonl

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} run_dir=${RUN_DIR} init=${INIT_FROM}"
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
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
  --cond-adanorm --cond-drop "${COND_DROP}"
  --coord-value-emb
  --noise-profile "${NOISE_PROFILE}"
  --lazy-data auto
  --init-from "${INIT_FROM}"
  --init-reset-spectra
)

echo "=== prepare: tokenizer + lazy index caches $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
