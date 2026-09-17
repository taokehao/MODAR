#!/bin/bash
# Offline Rietveld-lite (anisotropic lattice refine) on frozen noisy pools.
#SBATCH --job-name=rietveld_lite
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rietveld_lite_%j.out
#SBATCH --error=logs/rietveld_lite_%j.err

set -euo pipefail
cd /home/tkh/python-project/multimodal_csp-v2
mkdir -p logs
PY=/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python

echo "start=$(date) host=$(hostname)"

# Base: high-info free rungs under noisy FT pool
"${PY}" crystal_language/refine_rerank_ablation.py \
  --eval-dir crystal_language/eval_results/ladder_tierA_s2_noisy_exp_500 \
  --data-dir data_prep/outputs/crystal_language_unified_tierA_v2 \
  --conditions pxrd_xrf,pxrd_neutron_xrf,pxrd_neutron_xrf_xps,pxrd_formula \
  --lattices 0 \
  --refine-top-m 25 \
  --scale-max 0.03 --n-scales 7 \
  --axis-max 0.025 --n-axis 5 \
  --match-sigma 0.25 --shift-max 0.25 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --out crystal_language/eval_results/refine_rerank_ablation_base_free.json

# Bigcell free (structures stored in rraware run)
"${PY}" crystal_language/refine_rerank_ablation.py \
  --eval-dir crystal_language/eval_results/bigcell_ab/noisy_bigcell_free_rraware \
  --data-dir crystal_language/eval_results/bigcell_ab/data \
  --split test_big20 \
  --conditions pxrd_neutron_xrf \
  --lattices 0 \
  --refine-top-m 25 \
  --scale-max 0.03 --n-scales 7 \
  --axis-max 0.025 --n-axis 5 \
  --match-sigma 0.25 --shift-max 0.25 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --out crystal_language/eval_results/refine_rerank_ablation_bigcell_free.json

echo "done=$(date)"
