#!/usr/bin/env python
"""Assemble the multimodal condition ladder into one table.

Merges the per-shard evaluation details produced by
``submit_eval_ladder_realpxrd_array.sbatch`` for every rung x lattice mode and
emits a consolidated JSON + a markdown table. Optionally joins a metadata
index (by ``id``) to also report Top-1 broken down by crystal system, which is
where the neutron / XPS modalities are expected to add the most.

Usage
  python crystal_language/merge_ladder.py \
    --eval-dir crystal_language/eval_results/ladder_realpxrd_s1 \
    --split test_shuffled \
    [--metadata-index data_prep/outputs/metadata_index_realpxrd/metadata_index.jsonl.gz]
"""

from __future__ import annotations

import argparse
import glob as globlib
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

RUNGS = [
    "pxrd",
    "pxrd_element",
    "pxrd_xrf",
    "pxrd_neutron",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
]
TOP_KS = [1, 5, 10, 20, 50, 100]


def topk_any(cands: list[dict[str, Any]], key: str, k: int) -> bool:
    return any(bool(c.get(key)) for c in cands[:k])


def load_index_systems(path: Path) -> dict[str, str]:
    """Map id -> crystal_system from a metadata index (jsonl.gz or parquet)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            t = pq.read_table(path, columns=["id", "crystal_system"])
            for i, sysname in zip(t.column("id").to_pylist(), t.column("crystal_system").to_pylist()):
                out[str(i)] = str(sysname)
        except Exception:
            return out
        return out
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["id"])] = str(row.get("crystal_system") or "unknown")
    return out


def merge_config(eval_dir: Path, split: str, cond: str, lat: int) -> dict[str, Any]:
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr*_start*_n*_s*.jsonl"
    paths = [p for p in globlib.glob(str(eval_dir / pattern)) if "_summary" not in p]
    paths.sort(key=lambda p: int(re.search(r"start(\d+)", p).group(1)) if re.search(r"start(\d+)", p) else 0)

    seen: set[int] = set()
    evaluated = 0
    comp_records = 0
    top_s: dict[int, int] = {k: 0 for k in TOP_KS}
    top_rr: dict[int, int] = {k: 0 for k in TOP_KS}
    by_sys: dict[str, Counter] = {}
    top1_by_id: dict[str, bool] = {}

    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "target_error" in row:
                    continue
                idx = row.get("index")
                if idx is not None:
                    if idx in seen:
                        continue
                    seen.add(idx)
                cands = row.get("candidates") or []
                evaluated += 1
                comp_records += int(topk_any(cands, "composition_match", max(TOP_KS)))
                for k in TOP_KS:
                    top_s[k] += int(topk_any(cands, "structure_match", k))
                order = row.get("rerank_order")
                reranked = [cands[i] for i in order if i < len(cands)] if order else cands
                for k in TOP_KS:
                    top_rr[k] += int(topk_any(reranked, "structure_match", k))
                rid = row.get("id")
                if rid is not None:
                    top1_by_id[str(rid)] = topk_any(reranked, "structure_match", 1)

    return {
        "condition_set": cond,
        "use_lattice": lat,
        "n_records": evaluated,
        "shards": len(paths),
        "comp_rate": comp_records / max(1, evaluated),
        "topk_sampling": {str(k): top_s[k] / max(1, evaluated) for k in TOP_KS},
        "topk_rerank": {str(k): top_rr[k] / max(1, evaluated) for k in TOP_KS},
        "_top1_by_id": top1_by_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--split", default="test_shuffled")
    parser.add_argument("--metadata-index", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--rungs",
        default=",".join(RUNGS),
        help="Comma-separated rung list to merge/print; defaults to the full ladder.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rungs = [x.strip() for x in args.rungs.split(",") if x.strip()]
    id_to_system = {}
    if args.metadata_index:
        id_to_system = load_index_systems(Path(args.metadata_index))

    table: dict[str, dict[str, Any]] = {"lattice_free": {}, "known_lattice": {}}
    by_system_rows: list[dict[str, Any]] = []
    for lat, name in ((0, "lattice_free"), (1, "known_lattice")):
        for cond in rungs:
            res = merge_config(eval_dir, args.split, cond, lat)
            top1_by_id = res.pop("_top1_by_id")
            table[name][cond] = res
            if id_to_system:
                sys_top1: dict[str, list[int]] = {}
                for rid, top1_hit in top1_by_id.items():
                    sysname = id_to_system.get(rid, "unknown")
                    bucket = sys_top1.setdefault(sysname, [0, 0])
                    bucket[1] += 1
                    bucket[0] += int(top1_hit)
                for sysname, (hit, n) in sorted(sys_top1.items()):
                    by_system_rows.append({
                        "lattice": name, "condition_set": cond,
                        "crystal_system": sysname, "n": n,
                        "top1_rerank": hit / max(1, n),
                    })

    out_path = Path(args.out) if args.out else eval_dir / f"ladder_{args.split}_table.json"
    out_path.write_text(
        json.dumps({"table": table, "by_crystal_system": by_system_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def fmt(x: float) -> str:
        return f"{100 * x:5.1f}"

    print(f"\n=== Multimodal ladder ({args.split}) — rerank Top-k (%) ===\n")
    for name in ("lattice_free", "known_lattice"):
        print(f"[{name}]")
        print(f"{'rung':<26}{'N':>5}{'comp':>7}{'T1':>7}{'T5':>7}{'T10':>7}{'T20':>7}{'T50':>7}")
        for cond in rungs:
            r = table[name][cond]
            rr = r["topk_rerank"]
            print(f"{cond:<26}{r['n_records']:>5}{fmt(r['comp_rate']):>7}"
                  f"{fmt(rr['1']):>7}{fmt(rr['5']):>7}{fmt(rr['10']):>7}"
                  f"{fmt(rr['20']):>7}{fmt(rr['50']):>7}")
        print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
