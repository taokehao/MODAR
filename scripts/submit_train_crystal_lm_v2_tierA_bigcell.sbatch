#!/bin/bash
#SBATCH --job-name=crylm_bigcell
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4090D:6
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/crystal_lm_v2_tierA_bigcell_%j.out
#SBATCH --error=logs/crystal_lm_v2_tierA_bigcell_%j.err

# Large-cell specialist (plan section: 专用大晶胞模型 / 纯专家 + 热启动).
#
# Motivation: >20-atom cells are the sole remaining frontier (small cells already
# reach/exceed the PXRD+formula oracle). In the mix Tier-A model ~89% of training
# signal comes from <=20-atom cells, so large cells are chronically under-served.
# This run keeps ONLY n_sites>=21 (362,673 train / 3,616 valid records) and warm-
# starts from the validated Tier-A S2 checkpoint, spending the whole capacity on
# large cells without the forgetting/regularization tradeoff that limited RL.
#
# Same validated V2 architecture as tierA (S2 16L/1024d/16H + AdaLN + coord value
# emb + mix condition + mix lattice); block_size stays 1024 (large cells can reach
# ~660 target tokens, so shrinking the block would truncate exactly the hardest
# samples this specialist targets). Inference-time routing: n_sites<=20 -> base
# tierA model, n_sites>20 -> this specialist.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

NGPU=${NGPU:-6}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_bigcell}
# Warm-start source: the validated full-distribution Tier-A S2 model.
BASE_RUN=${BASE_RUN:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
INIT_FROM=${INIT_FROM:-${BASE_RUN}/checkpoint_last.pt}

MIN_SITES=${MIN_SITES:-21}
CONDITION_SET=${CONDITION_SET:-mix}
USE_LATTICE=${USE_LATTICE:-mix}
TARGET_MODE=${TARGET_MODE:-full}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
# S2 scale (must match the warm-start checkpoint exactly).
N_LAYER=${N_LAYER:-16}
N_HEAD=${N_HEAD:-16}
N_EMBD=${N_EMBD:-1024}
# Effective batch 16*2*6 = 192 (== validated Tier-A recipe).
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
# Fine-tuning LR (below the 3e-4 pretrain LR) to preserve warm-started geometry.
LR=${LR:-1e-4}
# ~8 epochs over 362,673 train: 8*362673 / 192 ~= 15000 steps.
MAX_STEPS=${MAX_STEPS:-15000}
WARMUP_STEPS=${WARMUP_STEPS:-500}
EVAL_EVERY=${EVAL_EVERY:-1000}
SAVE_EVERY=${SAVE_EVERY:-1500}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-bf16}
COND_DROP=${COND_DROP:-0.1}
NOISE_PROFILE=${NOISE_PROFILE:-none}

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
  echo "FATAL: warm-start checkpoint missing: ${INIT_FROM}" >&2
  exit 1
fi
# --init-from requires the matching tokenizer.json in the new out_dir.
if [[ ! -s "${RUN_DIR}/tokenizer.json" ]]; then
  cp "${BASE_RUN}/tokenizer.json" "${RUN_DIR}/tokenizer.json"
  echo "copied tokenizer.json from ${BASE_RUN} into ${RUN_DIR}"
fi
ls -lh "${DATA_DIR}"/*.jsonl
echo "warm-start from ${INIT_FROM}  min_sites=${MIN_SITES}"

export OMP_NUM_THREADS=8
echo "node=$(hostname) ngpu=${NGPU} run_dir=${RUN_DIR} data=${DATA_DIR}"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

COMMON_ARGS=(
  --arch v2 --rep v2
  --data-dir "${DATA_DIR}"
  --out-dir "${RUN_DIR}"
  --init-from "${INIT_FROM}"
  --min-sites "${MIN_SITES}"
  --condition-set "${CONDITION_SET}"
  --use-lattice "${USE_LATTICE}"
  --target-mode "${TARGET_MODE}"
  --block-size "${BLOCK_SIZE}"
  --n-layer "${N_LAYER}" --n-head "${N_HEAD}" --n-embd "${N_EMBD}"
  --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM}"
  --lr "${LR}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}"
  --eval-every "${EVAL_EVERY}" --save-every "${SAVE_EVERY}"
  --num-workers "${NUM_WORKERS}" --amp "${AMP}"
  --cond-adanorm --cond-drop "${COND_DROP}"
  --coord-value-emb
  --noise-profile "${NOISE_PROFILE}"
  --lazy-data auto
)

echo "=== prepare: tokenizer + lazy index caches (min_sites=${MIN_SITES}) $(date) ==="
srun python -u crystal_language/train.py "${COMMON_ARGS[@]}" --prepare-only --device cpu

echo "=== train $(date) ==="
srun python -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" \
  crystal_language/train.py "${COMMON_ARGS[@]}"

echo "DONE $(date)"
