#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for the RealPXRD-Solver head-to-head.

Per target we have a binary Top-1 / Top-20 outcome for our pipeline (fusion
ranker over an N=100 pool, refined match) and for RPS (their released model run
on the same targets: Top-1 = single sample, Top-20 = any of 20 samples). The
difference in counts is resampled over targets (paired bootstrap, percentile
CI); McNemar's exact test gives a p-value on the discordant pairs. Our own rate
also gets a percentile CI so it can be read against RPS's published numbers,
which exist only as aggregates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rank_eval_pools import energy_scores, load_energy, load_pool, rr  # noqa: E402

REPORTED = {"cnrs": {"n": 126, "top1": 0.779, "top20": 0.919}, "rruff": {"n": 269, "top1": 0.788, "top20": 0.929}}


def our_hits(pool: Path, chgnet: Path, match_key: str) -> dict[str, tuple[bool, bool]]:
    recs = load_pool(pool)
    E = load_energy(chgnet)
    out = {}
    for rid, r in recs.items():
        c = r["candidates"]
        flags = np.array([bool(x.get(match_key, x.get("structure_match"))) for x in c])
        phys = np.array([(-1e9 if x.get("rerank_score") is None else float(x["rerank_score"])) for x in c])
        fused = rr(phys) + rr(energy_scores(c, E.get(rid, {})))
        order = np.argsort(-fused)
        out[rid] = (bool(flags[order[:1]].any()), bool(flags[order[:20]].any()))
    return out


def rps_hits(per_sample_dir: Path, prefix: str) -> dict[str, tuple[bool, bool]]:
    t1 = {f"{prefix}:{json.loads(l)['id']}": bool(json.loads(l)["match"]) for l in (per_sample_dir / "per_sample_n1.jsonl").open() if l.strip()}
    t20 = {f"{prefix}:{json.loads(l)['id']}": bool(json.loads(l)["match"]) for l in (per_sample_dir / "per_sample_n20.jsonl").open() if l.strip()}
    return {k: (t1[k], t20.get(k, False)) for k in t1}


def paired_ci(a: np.ndarray, b: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    diffs = (a[idx].sum(axis=1) - b[idx].sum(axis=1))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def rate_ci(a: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    s = a[idx].sum(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    if b01 + b10 == 0:
        return 1.0
    return float(binomtest(b01, b01 + b10, 0.5).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["cnrs", "rruff"], required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--chgnet", required=True)
    ap.add_argument("--rps-dir", required=True, help="realpxrd_solver_si/<set>_allL")
    ap.add_argument("--match-key", default="structure_match")
    ap.add_argument("--label", default="ours")
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(20260911)
    prefix = "rps_cnrs" if args.set == "cnrs" else "rps_rruff"
    ours = our_hits(Path(args.pool), Path(args.chgnet), args.match_key)
    theirs = rps_hits(Path(args.rps_dir), prefix)
    ids = sorted(set(ours) & set(theirs))
    if len(ids) != len(ours) or len(ids) != len(theirs):
        print(f"warning: paired on {len(ids)} targets (ours {len(ours)}, rps {len(theirs)})")
    o1 = np.array([ours[i][0] for i in ids], dtype=int)
    o20 = np.array([ours[i][1] for i in ids], dtype=int)
    r1 = np.array([theirs[i][0] for i in ids], dtype=int)
    r20 = np.array([theirs[i][1] for i in ids], dtype=int)
    n = len(ids)
    rep = REPORTED[args.set]
    res = {"set": args.set, "n": n, "label": args.label, "match_key": args.match_key}
    print(f"\n=== {args.set.upper()}  n={n}  {args.label} ({args.match_key}) vs RPS reproduced (paired) ===")
    for name, a, b, rep_rate in (("Top-1", o1, r1, rep["top1"]), ("Top-20", o20, r20, rep["top20"])):
        d_mean, lo, hi = paired_ci(a, b, args.B, rng)
        p = mcnemar_p(a, b)
        clo, chi = rate_ci(a, args.B, rng)
        rep_count = round(rep_rate * rep["n"])
        both = int(((a == 1) & (b == 1)).sum())
        only_o = int(((a == 1) & (b == 0)).sum())
        only_r = int(((a == 0) & (b == 1)).sum())
        print(
            f"  {name:6s} ours {a.sum():3d} ({100*a.mean():5.1f}%)  95% CI [{clo:.0f}, {chi:.0f}]   "
            f"RPS reproduced {b.sum():3d} ({100*b.mean():5.1f}%)   Δ={a.sum()-b.sum():+d}  paired 95% CI [{lo:+.0f}, {hi:+.0f}]  McNemar p={p:.3f}"
            f"   (both {both}, only-ours {only_o}, only-RPS {only_r})"
        )
        print(f"         RPS reported {rep_count} ({100*rep_rate:.1f}%)  → ours − reported = {a.sum()-rep_count:+d}; reported {'inside' if clo <= rep_count <= chi else 'outside'} our CI")
        res[name] = {
            "ours": int(a.sum()), "ours_ci": [clo, chi], "rps_repro": int(b.sum()), "diff": int(a.sum() - b.sum()),
            "diff_ci": [lo, hi], "mcnemar_p": p, "rps_reported": rep_count, "both": both, "only_ours": only_o, "only_rps": only_r,
        }
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
