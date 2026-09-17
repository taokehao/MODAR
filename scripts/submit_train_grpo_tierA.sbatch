#!/bin/bash
# Stage-2 RL: GRPO + KL-to-base for the Tier-A crystal LM.
#
# One GPU, one rung (pxrd_neutron_xrf), big cells (>20 sites) by default.
# LATTICE_MODE=mix samples known-lattice prompts (teacher-forced cell,
# forced tokens masked out of the PG/KL sums) and lattice-free prompts from
# the same policy, so RL no longer trains only the lattice-free conditional.
#
#SBATCH --job-name=crylm_grpo
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090D:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/crystal_lm_grpo_%j.out
#SBATCH --error=logs/crystal_lm_grpo_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/tkh/python-project/multimodal_csp-v2}
CONDA_BASE=${CONDA_BASE:-/home/tkh/software/miniconda3}
CONDA_ENV=${CONDA_ENV:-pxrd_crystal_lm}

BASE_RUN=${BASE_RUN:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_ada_cve}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data_prep/outputs/crystal_language_unified_tierA_v2}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/crystal_language/runs/tierA_s2_grpo2}

SPLIT=${SPLIT:-train}
COND=${COND:-pxrd_neutron_xrf}
# free / known / mix; mix draws known-lattice prompts with prob KNOWN_FRAC.
LATTICE_MODE=${LATTICE_MODE:-mix}
KNOWN_FRAC=${KNOWN_FRAC:-0.5}
START_INDEX=${START_INDEX:-0}
# 256 records x 768 steps @ known_frac 0.5 = ~1.5 passes/branch, matching the
# big1 recipe that produced the proven +4pp (per-branch step budget == big1).
MAX_RECORDS=${MAX_RECORDS:-256}
MAX_STEPS=${MAX_STEPS:-768}
GROUP_SIZE=${GROUP_SIZE:-8}
LR=${LR:-2e-6}
KL_BETA=${KL_BETA:-0.02}
# 0 = plain adv/std (big1 behavior). A positive floor neutered the dense
# physical-score signal in GRPO-2 and caused a broad regression; keep it off.
ADV_STD_FLOOR=${ADV_STD_FLOOR:-0.0}
TEMP=${TEMP:-0.8}
TOP_K=${TOP_K:-20}
AMP=${AMP:-bf16}
# Focus on large cells (>20 atoms) where reward variance / RL signal is highest.
MIN_SITES=${MIN_SITES:-21}
MAX_SITES=${MAX_SITES:-100000}
SCAN_LIMIT=${SCAN_LIMIT:-200000}
LOG_SAMPLES=${LOG_SAMPLES:-1}
# Big default so only the final checkpoint is written (each is ~1.2 GB); override
# to snapshot intermediates.
SAVE_EVERY=${SAVE_EVERY:-1000000}

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

# Reuse the base tokenizer so later eval scripts can load the GRPO run dir.
if [[ ! -f "${OUT_DIR}/tokenizer.json" ]]; then
  cp "${BASE_RUN}/tokenizer.json" "${OUT_DIR}/tokenizer.json"
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "node=$(hostname) base=${BASE_RUN} out=${OUT_DIR} cond=${COND} lattice=${LATTICE_MODE}(known_frac=${KNOWN_FRAC}) steps=${MAX_STEPS} group=${GROUP_SIZE} sites=[${MIN_SITES},${MAX_SITES}]"
python -c "import torch; print('torch', torch.__version__, 'ngpu', torch.cuda.device_count())"

LOG_SAMPLES_ARG=()
if [[ "${LOG_SAMPLES}" == "1" ]]; then
  LOG_SAMPLES_ARG=(--log-samples)
fi

srun python -u crystal_language/train_grpo.py \
  --run-dir "${BASE_RUN}" \
  --checkpoint checkpoint_last.pt \
  --data-dir "${DATA_DIR}" \
  --split "${SPLIT}" \
  --out-dir "${OUT_DIR}" \
  --condition-set "${COND}" \
  --lattice-mode "${LATTICE_MODE}" \
  --known-frac "${KNOWN_FRAC}" \
  --start-index "${START_INDEX}" \
  --max-records "${MAX_RECORDS}" \
  --min-sites "${MIN_SITES}" \
  --max-sites "${MAX_SITES}" \
  --scan-limit "${SCAN_LIMIT}" \
  --max-steps "${MAX_STEPS}" \
  --group-size "${GROUP_SIZE}" \
  --samples-per-batch "${GROUP_SIZE}" \
  --temperature "${TEMP}" \
  --top-k-sampling "${TOP_K}" \
  --lr "${LR}" \
  --kl-beta "${KL_BETA}" \
  --adv-std-floor "${ADV_STD_FLOOR}" \
  --save-every "${SAVE_EVERY}" \
  "${LOG_SAMPLES_ARG[@]}" \
  --amp "${AMP}"

echo "DONE $(date)"
