#!/usr/bin/env bash
# Refine-before-match over whole candidate pools, one shard file per process, spread over
# leftover GPUs (idle cards; VASP-resident cards are fine for these small relaxations).
# Usage: run_refine_pools.sh <pool_dir> <native_jsonl> <out_dir> "<gpu list>" [max_relax]
set -u
ROOT=/home/tkh/python-project/multimodal_csp-v2
PY=/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python
POOL=$1; NATIVE=$2; OUT=$3; GPUS=($4); MAXR=${5:-40}
mkdir -p "$OUT" "$ROOT/logs"
FILES=($(ls "$POOL"/test_pxrd_formula_lat1_*_s100.jsonl))
NG=${#GPUS[@]}
for ((g=0; g<NG; g++)); do
  (
    for ((i=g; i<${#FILES[@]}; i+=NG)); do
      f=${FILES[$i]}; base=$(basename "$f" .jsonl)
      [ -f "$OUT/$(basename "$f")" ] && continue
      CUDA_VISIBLE_DEVICES=${GPUS[$g]} OMP_NUM_THREADS=4 "$PY" -u "$ROOT/scripts/refine_match_candidates.py" \
        --eval-dir "$POOL" --native "$NATIVE" --out-dir "$OUT" --glob "$(basename "$f")" --max-relax "$MAXR" \
        > "$ROOT/logs/refine_$(basename "$OUT")_${base}_gpu${GPUS[$g]}.log" 2>&1
    done
    echo "gpu ${GPUS[$g]} queue done"
  ) &
done
wait
echo "REFINE-POOL-DONE $OUT"
