#!/usr/bin/env python
"""Merge sharded evaluation details JSONL files into one summary.

Recomputes candidate rates and Top-k metrics (sampling order and reranked)
from the per-record rows, so shards of unequal size merge correctly.

Usage:
  python crystal_language/merge_eval_results.py \
      --glob 'crystal_language/eval_results/RUN/valid_..._cd1_rr1_start*_s100.jsonl' \
      --out  crystal_language/eval_results/RUN/valid_..._cd1_rr1_merged_summary.json
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def topk_any(candidates: list[dict[str, Any]], key: str, k: int) -> bool:
    return any(bool(c.get(key)) for c in candidates[:k])


def topk_best_rmsd(candidates: list[dict[str, Any]], k: int) -> float | None:
    vals = [
        c["rmsd"]
        for c in candidates[:k]
        if c.get("structure_match") and c.get("rmsd") is not None
    ]
    return min(vals) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", required=True, help="shard details JSONL glob")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-ks", default="1,20,50,100")
    args = parser.parse_args()

    paths = sorted(
        globlib.glob(args.glob),
        key=lambda p: int(re.search(r"start(\d+)", p).group(1)) if re.search(r"start(\d+)", p) else 0,
    )
    if not paths:
        raise SystemExit(f"no files match {args.glob}")

    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    counts: Counter[str] = Counter()
    top_counts: dict[int, Counter[str]] = {k: Counter() for k in top_ks}
    top_counts_rr: dict[int, Counter[str]] = {k: Counter() for k in top_ks}
    rmse_sums: dict[int, float] = {k: 0.0 for k in top_ks}
    rmse_counts: dict[int, int] = {k: 0 for k in top_ks}
    rmse_sums_rr: dict[int, float] = {k: 0.0 for k in top_ks}
    rmse_counts_rr: dict[int, int] = {k: 0 for k in top_ks}
    rank_dist: Counter[str] = Counter()
    rank_dist_rr: Counter[str] = Counter()
    n_sites_buckets: dict[str, Counter[str]] = {}
    any_rerank = False
    seen_indices: set[int] = set()

    def rank_bucket(cands: list[dict[str, Any]]) -> str:
        for i, c in enumerate(cands):
            if c.get("structure_match"):
                if i == 0:
                    return "rank_1"
                if i < 20:
                    return "rank_2_20"
                if i < 50:
                    return "rank_21_50"
                return "rank_51_plus"
        return "no_match"

    def sites_bucket(n: int) -> str:
        if n <= 8:
            return "n<=8"
        if n <= 20:
            return "9-20"
        if n <= 40:
            return "21-40"
        return ">40"

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = row.get("index")
                if idx is not None:
                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)
                if "target_error" in row:
                    counts["records"] += 1
                    counts["target_parse_failed"] += 1
                    continue
                cands = row.get("candidates") or []
                counts["records"] += 1
                for c in cands:
                    counts["candidates"] += 1
                    counts["parse_valid"] += int(bool(c.get("parse_valid")))
                    counts["composition_match"] += int(bool(c.get("composition_match")))
                    counts["structure_match"] += int(bool(c.get("structure_match")))
                for k in top_ks:
                    top_counts[k]["parse_success"] += int(topk_any(cands, "parse_valid", k))
                    top_counts[k]["composition_success"] += int(
                        topk_any(cands, "composition_match", k)
                    )
                    top_counts[k]["structure_success"] += int(
                        topk_any(cands, "structure_match", k)
                    )
                    best = topk_best_rmsd(cands, k)
                    if best is not None:
                        rmse_sums[k] += best
                        rmse_counts[k] += 1
                rank_dist[rank_bucket(cands)] += 1

                bucket = sites_bucket(int(row.get("target_n_sites") or 0))
                b = n_sites_buckets.setdefault(bucket, Counter())
                b["records"] += 1
                b["top1"] += int(topk_any(cands, "structure_match", 1))
                b["topmax"] += int(topk_any(cands, "structure_match", max(top_ks)))

                order = row.get("rerank_order")
                if order:
                    any_rerank = True
                    reranked = [cands[i] for i in order if i < len(cands)]
                    for k in top_ks:
                        top_counts_rr[k]["structure_success"] += int(
                            topk_any(reranked, "structure_match", k)
                        )
                        top_counts_rr[k]["composition_success"] += int(
                            topk_any(reranked, "composition_match", k)
                        )
                        top_counts_rr[k]["parse_success"] += int(
                            topk_any(reranked, "parse_valid", k)
                        )
                        best = topk_best_rmsd(reranked, k)
                        if best is not None:
                            rmse_sums_rr[k] += best
                            rmse_counts_rr[k] += 1
                    rank_dist_rr[rank_bucket(reranked)] += 1

    evaluated = counts["records"] - counts["target_parse_failed"]
    summary: dict[str, Any] = {
        "records": counts["records"],
        "evaluated": evaluated,
        "candidate_counts": {
            "candidates": counts["candidates"],
            "parse": counts["parse_valid"],
            "comp": counts["composition_match"],
            "struct": counts["structure_match"],
        },
        "candidate_rates": {
            "parse": counts["parse_valid"] / max(1, counts["candidates"]),
            "comp": counts["composition_match"] / max(1, counts["candidates"]),
            "struct": counts["structure_match"] / max(1, counts["candidates"]),
        },
        "topk_rates": {
            str(k): {
                "parse": top_counts[k]["parse_success"] / max(1, evaluated),
                "composition": top_counts[k]["composition_success"] / max(1, evaluated),
                "structure": top_counts[k]["structure_success"] / max(1, evaluated),
                "rmse": (rmse_sums[k] / rmse_counts[k]) if rmse_counts[k] else None,
                "rmse_n": rmse_counts[k],
            }
            for k in top_ks
        },
        "match_rank_distribution": dict(rank_dist),
        "by_n_sites": {
            bucket: {
                "records": c["records"],
                "top1": c["top1"] / max(1, c["records"]),
                f"top{max(top_ks)}": c["topmax"] / max(1, c["records"]),
            }
            for bucket, c in sorted(n_sites_buckets.items())
        },
        "source_files": paths,
    }
    if any_rerank:
        summary["topk_rates_reranked"] = {
            str(k): {
                "parse": top_counts_rr[k]["parse_success"] / max(1, evaluated),
                "composition": top_counts_rr[k]["composition_success"] / max(1, evaluated),
                "structure": top_counts_rr[k]["structure_success"] / max(1, evaluated),
                "rmse": (rmse_sums_rr[k] / rmse_counts_rr[k]) if rmse_counts_rr[k] else None,
                "rmse_n": rmse_counts_rr[k],
            }
            for k in top_ks
        }
        summary["match_rank_distribution_reranked"] = dict(rank_dist_rr)

    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
