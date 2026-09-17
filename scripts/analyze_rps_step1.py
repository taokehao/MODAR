#!/usr/bin/env python3
"""Fuse physics rerank with MLIP energy; optionally merge extra adaptive-N samples.

    fused = rr(physics_rerank_score) + rr(-e_per_atom)   # k=10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RPS = {
    "cnrs": {"n": 126, "t1": 101, "t20": 118},
    "rruff": {"n": 269, "t1": 199, "t20": 245},
}


def rr(values: np.ndarray, k: float = 10.0) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float))
    rank = np.empty(len(values), dtype=float)
    rank[order] = np.arange(len(values))
    return 1.0 / (k + rank)


def load_eval(eval_dir: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    paths = sorted(eval_dir.glob("test_pxrd_formula_lat1_*_s100.jsonl"))
    if not paths:
        paths = [
            p
            for p in sorted(eval_dir.glob("test_pxrd_formula_lat1_*.jsonl"))
            if not p.name.endswith("_summary.json")
        ]
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                recs.setdefault(row["id"], row)
    return recs


def load_energy(path: Path | None) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    if path is None or not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["id"]] = {int(s["cand"]): float(s["e_per_atom"]) for s in row["scores"]}
    return out


def merge_extra(
    base: dict[str, dict],
    extra: dict[str, dict],
    extra_energy: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    """Append extra candidates; remap extra energy indices by the base pool size."""
    remapped: dict[str, dict[int, float]] = {}
    for rid, row in extra.items():
        if rid not in base:
            base[rid] = row
            remapped[rid] = dict(extra_energy.get(rid, {}))
            continue
        offset = len(base[rid].get("candidates") or [])
        base[rid]["candidates"] = list(base[rid].get("candidates") or []) + list(
            row.get("candidates") or []
        )
        remapped[rid] = {offset + i: e for i, e in extra_energy.get(rid, {}).items()}
    return remapped


def hit_at(flags: np.ndarray, scores: np.ndarray, k: int) -> bool:
    order = np.argsort(-scores)
    return bool(flags[order[:k]].any())


def summarize(
    recs: dict[str, dict],
    energy: dict[str, dict[int, float]],
    k_rr: float = 10.0,
) -> dict:
    n = len(recs)
    counts = {
        "n": n,
        "oracle": 0,
        "s@1": 0,
        "rr@1": 0,
        "rr@20": 0,
        "energy@1": 0,
        "energy@20": 0,
        "fusion@1": 0,
        "fusion@20": 0,
        "scored": 0,
        "n_small": 0,
        "n_large": 0,
        "oracle_small": 0,
        "oracle_large": 0,
        "fusion@1_small": 0,
        "fusion@1_large": 0,
        "fusion@20_small": 0,
        "fusion@20_large": 0,
    }
    for rid, row in recs.items():
        cands = row.get("candidates") or []
        flags = np.array([bool(c.get("structure_match")) for c in cands], dtype=bool)
        phys = np.array(
            [(-1e9 if c.get("rerank_score") is None else float(c["rerank_score"])) for c in cands],
            dtype=float,
        )
        n_sites = int(row.get("target_n_sites") or 0)
        large = n_sites >= 13
        if large:
            counts["n_large"] += 1
        else:
            counts["n_small"] += 1
        if flags.any():
            counts["oracle"] += 1
            counts["oracle_large" if large else "oracle_small"] += 1
        if flags[:1].any():
            counts["s@1"] += 1
        if hit_at(flags, phys, 1):
            counts["rr@1"] += 1
        if hit_at(flags, phys, 20):
            counts["rr@20"] += 1
        e_map = energy.get(rid, {})
        energy_arr = np.full(len(cands), np.nan, dtype=float)
        for idx, val in e_map.items():
            if 0 <= idx < len(cands):
                energy_arr[idx] = val
        if np.isfinite(energy_arr).any():
            counts["scored"] += 1
            e_score = np.where(np.isfinite(energy_arr), -energy_arr, -1e9)
            fused = rr(phys, k=k_rr) + rr(e_score, k=k_rr)
        else:
            e_score = phys
            fused = phys
        if hit_at(flags, e_score, 1):
            counts["energy@1"] += 1
        if hit_at(flags, e_score, 20):
            counts["energy@20"] += 1
        f1 = hit_at(flags, fused, 1)
        f20 = hit_at(flags, fused, 20)
        if f1:
            counts["fusion@1"] += 1
            counts["fusion@1_large" if large else "fusion@1_small"] += 1
        if f20:
            counts["fusion@20"] += 1
            counts["fusion@20_large" if large else "fusion@20_small"] += 1
    return counts


def fmt(counts: dict, key: str) -> str:
    n = counts["n"]
    return f"{counts[key]:3d}/{n} ({100 * counts[key] / n:5.1f}%)"


def print_block(title: str, counts: dict, rps_key: str | None) -> None:
    print(f"\n=== {title}  n={counts['n']} scored={counts['scored']} ===")
    print(f"  oracle     {fmt(counts, 'oracle')}")
    print(f"  s@1        {fmt(counts, 's@1')}")
    print(f"  physics@1  {fmt(counts, 'rr@1')}")
    print(f"  energy@1   {fmt(counts, 'energy@1')}")
    print(f"  fusion@1   {fmt(counts, 'fusion@1')}")
    print(f"  physics@20 {fmt(counts, 'rr@20')}")
    print(f"  energy@20  {fmt(counts, 'energy@20')}")
    print(f"  fusion@20  {fmt(counts, 'fusion@20')}")
    if counts["n_large"]:
        print(
            f"  large n>=13  oracle {counts['oracle_large']}/{counts['n_large']}"
            f"  fusion@1 {counts['fusion@1_large']}/{counts['n_large']}"
            f"  fusion@20 {counts['fusion@20_large']}/{counts['n_large']}"
        )
        print(
            f"  small n<13   oracle {counts['oracle_small']}/{counts['n_small']}"
            f"  fusion@1 {counts['fusion@1_small']}/{counts['n_small']}"
            f"  fusion@20 {counts['fusion@20_small']}/{counts['n_small']}"
        )
    if rps_key:
        ref = RPS[rps_key]
        d1 = counts["fusion@1"] - ref["t1"]
        d20 = counts["fusion@20"] - ref["t20"]
        print(f"  vs RPS T1={ref['t1']}  Δfusion@1={d1:+d}   T20={ref['t20']}  Δfusion@20={d20:+d}")


def combine_energy(*maps: dict[str, dict[int, float]]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for mp in maps:
        for rid, items in mp.items():
            out.setdefault(rid, {}).update(items)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-eval", required=True)
    ap.add_argument("--extra-eval", default="")
    ap.add_argument("--base-energy", default="")
    ap.add_argument("--extra-energy", default="")
    ap.add_argument("--rps", choices=["cnrs", "rruff", ""], default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    recs = load_eval(Path(args.base_eval))
    base_e = load_energy(Path(args.base_energy) if args.base_energy else None)
    extra_e_remap: dict[str, dict[int, float]] = {}
    if args.extra_eval:
        extra = load_eval(Path(args.extra_eval))
        extra_e = load_energy(Path(args.extra_energy) if args.extra_energy else None)
        extra_e_remap = merge_extra(recs, extra, extra_e)
    energy = combine_energy(base_e, extra_e_remap)
    counts = summarize(recs, energy)
    print_block(args.name, counts, args.rps or None)
    if args.out:
        Path(args.out).write_text(json.dumps(counts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
