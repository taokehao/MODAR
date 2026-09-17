#!/bin/bash
#SBATCH --job-name=noisy_rr_ablat
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/noisy_rerank_ablation_%j.out
#SBATCH --error=logs/noisy_rerank_ablation_%j.err

set -euo pipefail
cd /home/tkh/python-project/multimodal_csp-v2
mkdir -p logs
PY=/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python

echo "node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date)"
"${PY}" crystal_language/noisy_rerank_ablation.py spectrum \
  --eval-dir crystal_language/eval_results/ladder_tierA_s2_noisy_exp_500 \
  --data-dir data_prep/outputs/crystal_language_unified_tierA_v2 \
  --conditions pxrd_element,pxrd_xrf,pxrd_neutron_xrf,pxrd_neutron_xrf_xps,pxrd_formula \
  --lattices 0 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --out crystal_language/eval_results/noisy_rerank_ablation_spectrum.json

echo "done=$(date)"
