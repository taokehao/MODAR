#!/bin/bash
#SBATCH --job-name=crylm_rps_lat
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4090D:4
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/crystal_lm_v2_tierA_di_rps_formula_lat_%j.out
#SBATCH --error=logs/crystal_lm_v2_tierA_di_rps_formula_lat_%j.err

# RPS comparison specialist. Does NOT replace the paper mix model.
#
# Paper tables stay on crystal_language/runs/tierA_s2_di (mix conditions,
# lattice_free_prob=0.5). This run is a separate head used only vs RealPXRD-Solver
# under the shared protocol: PXRD + formula + known lattice.
#
# Data: the same 3,456,202 Tier-A records. Every update is
#   condition_set=pxrd_formula, use_lattice=always.
# Warm-start the 386M d–I decoder+encoder from tierA_s2_di (do not reset spectra).
# ~20k steps at batch 192 is a bit over one epoch of protocol-matched data,
# vs ~1/14 of mix updates that saw this protocol during the original d–I FT.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-4}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_di_rps_formula_lat}
BASE_RUN=${BASE_RUN:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_di}
INIT_FROM=${INIT_FROM:-${BASE_RUN}/checkpoint_last.pt}

CONDITION_SET=${CONDITION_SET:-pxrd_formula}
USE_LATTICE=${USE_LATTICE:-always}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
BATCH_SIZE=${BATCH_SIZE:-12}
GRAD_ACCUM=${GRAD_ACCUM:-4}
# Decoder and d–I encoder are both already trained; keep FT LR below the 1e-4
# used when the spectra encoder was randomly initialized.
LR=${LR:-5e-5}
MAX_STEPS=${MAX_STEPS:-20000}
WARMUP_STEPS=${WARMUP_STEPS:-500}
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
echo "protocol: condition=${CONDITION_SET} use_lattice=${USE_LATTICE} (paper mix run untouched: ${BASE_RUN})"
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
)

echo "=== prepare: tokenizer + lazy index caches $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
