#!/usr/bin/env python3
"""Rank-fuse physics rerank scores with CHGNet energy.

Same protocol as the earlier SI probes: reciprocal rank with k=10, no fitted
weights.

    fused = rr(physics) + rr(-energy_per_atom)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rr(values: np.ndarray, k: float = 10.0) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float))
    rank = np.empty(len(values), dtype=float)
    rank[order] = np.arange(len(values))
    return 1.0 / (k + rank)


def load_eval(eval_dir: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for path in sorted(eval_dir.glob("test_pxrd_formula_lat1_*_s100.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                recs.setdefault(row["id"], row)
    return recs


def load_chgnet(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[row["id"]] = {int(s["cand"]): float(s["e_per_atom"]) for s in row["scores"]}
    return out


def hit_at(flags: np.ndarray, scores: np.ndarray, k: int) -> bool:
    order = np.argsort(-scores)
    return bool(flags[order[:k]].any())


def summarize(name: str, eval_dir: Path, chgnet_path: Path, k_rr: float) -> dict:
    recs = load_eval(eval_dir)
    chg = load_chgnet(chgnet_path)
    n = len(recs)
    counts = {
        "n": n,
        "chgnet_targets": 0,
        "oracle": 0,
        "s@1": 0,
        "rr@1": 0,
        "rr@20": 0,
        "energy@1": 0,
        "energy@20": 0,
        "fusion@1": 0,
        "fusion@10": 0,
        "fusion@20": 0,
        "fusion@50": 0,
    }
    missing_e = 0
    for rid, row in recs.items():
        cands = row.get("candidates") or []
        flags = np.array([bool(c.get("structure_match")) for c in cands], dtype=bool)
        phys = np.array(
            [(-1e9 if c.get("rerank_score") is None else float(c["rerank_score"])) for c in cands],
            dtype=float,
        )
        energy = np.full(len(cands), np.nan, dtype=float)
        for idx, e in chg.get(rid, {}).items():
            if 0 <= idx < len(cands):
                energy[idx] = e
        if flags.any():
            counts["oracle"] += 1
        if flags[:1].any():
            counts["s@1"] += 1
        if hit_at(flags, phys, 1):
            counts["rr@1"] += 1
        if hit_at(flags, phys, 20):
            counts["rr@20"] += 1
        if np.isfinite(energy).any():
            counts["chgnet_targets"] += 1
            e_score = np.where(np.isfinite(energy), -energy, -1e9)
            if hit_at(flags, e_score, 1):
                counts["energy@1"] += 1
            if hit_at(flags, e_score, 20):
                counts["energy@20"] += 1
            fused = rr(phys, k=k_rr) + rr(e_score, k=k_rr)
            for topk, key in ((1, "fusion@1"), (10, "fusion@10"), (20, "fusion@20"), (50, "fusion@50")):
                if hit_at(flags, fused, topk):
                    counts[key] += 1
        else:
            missing_e += 1
            if hit_at(flags, phys, 1):
                counts["fusion@1"] += 1
            if hit_at(flags, phys, 10):
                counts["fusion@10"] += 1
            if hit_at(flags, phys, 20):
                counts["fusion@20"] += 1
            if hit_at(flags, phys, 50):
                counts["fusion@50"] += 1

    rates = {key: (counts[key] / n if n else 0.0) for key in counts if key != "n"}
    print(f"\n=== {name}  n={n}  chgnet={counts['chgnet_targets']} missing_e={missing_e} ===")
    for key in ("s@1", "rr@1", "energy@1", "fusion@1", "fusion@10", "fusion@20", "oracle"):
        print(f"  {key:10s} {counts[key]:4d}/{n}  {100 * rates[key if key != 'n' else 'oracle']:5.1f}%")
    print(f"  rr@20      {counts['rr@20']:4d}/{n}  {100 * counts['rr@20'] / n:5.1f}%")
    print(f"  energy@20  {counts['energy@20']:4d}/{n}  {100 * counts['energy@20'] / n:5.1f}%")
    print(f"  fusion@50  {counts['fusion@50']:4d}/{n}  {100 * counts['fusion@50'] / n:5.1f}%")
    return {"name": name, "n": n, "k": k_rr, "counts": counts, "rates": rates}


def summarize_relax(name: str, eval_dir: Path, relax_path: Path) -> dict:
    recs = load_eval(eval_dir)
    relax: dict[str, list] = {}
    with relax_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            relax[row["id"]] = row["scores"]
    n = len(recs)
    hit_e = hit_match = oracle_after = scored = 0
    for rid, row in recs.items():
        flags = np.array(
            [bool(c.get("structure_match")) for c in (row.get("candidates") or [])],
            dtype=bool,
        )
        scores = relax.get(rid) or []
        if not scores:
            continue
        scored += 1
        e = np.array([float(s["e_relax"]) for s in scores], dtype=float)
        idx = [int(s["cand"]) for s in scores]
        best = int(np.argmin(e))
        cand = idx[best]
        if 0 <= cand < len(flags) and flags[cand]:
            hit_e += 1
        if any(bool(s.get("match_after")) for s in scores):
            oracle_after += 1
        if scores[best].get("match_after"):
            hit_match += 1
    print(f"\n=== {name} relax  scored={scored}/{n} ===")
    print(f"  min-E among relaxed (orig match): {hit_e}/{n}  {100 * hit_e / n:.1f}%")
    print(f"  min-E and match_after:          {hit_match}/{n}  {100 * hit_match / n:.1f}%")
    print(f"  any relaxed cand matches GT:   {oracle_after}/{n}  {100 * oracle_after / n:.1f}%")
    return {
        "name": name,
        "scored": scored,
        "n": n,
        "relax_energy@1": hit_e,
        "relax_match@1": hit_match,
        "oracle_after": oracle_after,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--relax", default="")
    args = ap.parse_args()
    root = Path("/home/tkh/python-project/multimodal_csp-v2")
    jobs = [
        (
            "CNRS SI profile",
            root / "crystal_language/eval_results/rps_cnrs126_si_profile_di",
            root / "crystal_language/eval_results/chgnet/rps_cnrs126_si_profile_di.jsonl",
        ),
        (
            "RRUFF SI peaks",
            root / "crystal_language/eval_results/rps_rruff269_si_peaks_di",
            root / "crystal_language/eval_results/chgnet/rps_rruff269_si_peaks_di.jsonl",
        ),
        (
            "RRUFF SI profile",
            root / "crystal_language/eval_results/rps_rruff269_si_profile_di",
            root / "crystal_language/eval_results/chgnet/rps_rruff269_si_profile_di.jsonl",
        ),
    ]
    reports = []
    for name, evd, chg in jobs:
        if not chg.exists() or not evd.exists():
            print(f"skip {name}: missing inputs")
            continue
        reports.append(summarize(name, evd, chg, args.k))
    if args.relax:
        reports.append(
            summarize_relax(
                "CNRS relax",
                root / "crystal_language/eval_results/rps_cnrs126_si_profile_di",
                Path(args.relax),
            )
        )
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
