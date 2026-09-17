#!/usr/bin/env python3
"""Condition ladder on the SI-encoded public experimental set (337 targets).

For every rung (condition set × lattice-free / known-lattice) report Top-1 /
Top-20 under the deployed ranker (physics + CHGNet fusion, k=10) before and
after refine-before-match, the raw physics-rerank numbers, the candidate-pool
oracle, and the n_sites<=20 subset. Paired bootstrap CIs are given for the
ladder steps that carry the paper's argument (PXRD → +XRF → +ND+XRF → formula).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rank_eval_pools import energy_scores, load_energy, load_pool, rr  # noqa: E402

CONDS = ["pxrd", "pxrd_xrf", "pxrd_neutron", "pxrd_neutron_xrf", "pxrd_neutron_xrf_xps", "pxrd_formula"]
LABEL = {
    "pxrd": "PXRD", "pxrd_xrf": "PXRD+XRF", "pxrd_neutron": "PXRD+ND", "pxrd_neutron_xrf": "PXRD+ND+XRF",
    "pxrd_neutron_xrf_xps": "PXRD+ND+XRF+XPS", "pxrd_formula": "PXRD+formula (reference)",
}
STEPS = [("pxrd", "pxrd_xrf"), ("pxrd_xrf", "pxrd_neutron_xrf"), ("pxrd_neutron_xrf", "pxrd_formula"), ("pxrd", "pxrd_neutron")]


def hits(pool_dir: Path, chgnet: Path | None, match_key: str) -> dict[str, dict]:
    recs = load_pool(pool_dir)
    E = load_energy(chgnet) if chgnet and chgnet.exists() else {}
    out = {}
    for rid, r in recs.items():
        c = r["candidates"]
        flags = np.array([bool(x.get(match_key, x.get("structure_match"))) for x in c])
        phys = np.array([(-1e9 if x.get("rerank_score") is None else float(x["rerank_score"])) for x in c])
        po = np.argsort(-phys)
        row = {
            "n_sites": int(r.get("target_n_sites") or 0), "oracle": bool(flags.any()),
            "phys1": bool(flags[po[:1]].any()), "phys20": bool(flags[po[:20]].any()),
        }
        if E:
            fo = np.argsort(-(rr(phys) + rr(energy_scores(c, E.get(rid, {})))))
            row["fus1"], row["fus20"] = bool(flags[fo[:1]].any()), bool(flags[fo[:20]].any())
        out[rid] = row
    return out


def ci_count(a: np.ndarray, rng, B=20000):
    idx = rng.integers(0, len(a), size=(B, len(a)))
    s = a[idx].sum(axis=1)
    return int(np.percentile(s, 2.5)), int(np.percentile(s, 97.5))


def paired(a: np.ndarray, b: np.ndarray, rng, B=20000):
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = b[idx].sum(axis=1) - a[idx].sum(axis=1)
    return int(b.sum() - a.sum()), int(np.percentile(d, 2.5)), int(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "crystal_language/eval_results/ladder_public_si"))
    ap.add_argument("--chgnet-dir", default=str(ROOT / "crystal_language/eval_results/chgnet"))
    ap.add_argument("--refined-suffix", default="_refined_cm", help="pool dir suffix holding refined matches")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root, chdir = Path(args.root), Path(args.chgnet_dir)
    rng = np.random.default_rng(20260911)
    table = {}
    for lat in (1, 0):
        print(f"\n##### public-337 SI, {'known lattice' if lat else 'lattice-free'} #####")
        print(f"  {'rung':26s} {'n':>4s} {'oracle':>6s} | {'phys@1':>6s} {'phys@20':>7s} | {'fus@1':>5s} {'fus@20':>6s} | {'ref@1':>5s} {'ref@20':>6s} | n<=20: {'ref@1':>5s} {'ref@20':>6s} {'oracle':>6s}")
        per = {}
        for cond in CONDS:
            d = root / f"{cond}_lat{lat}"
            if not d.exists():
                continue
            ch = chdir / f"ladder_public_si_{cond}_lat{lat}_chgnet.jsonl"
            raw = hits(d, ch, "structure_match")
            dref = Path(str(d) + args.refined_suffix)
            ref = hits(dref, ch, "structure_match") if dref.exists() else None
            ids = sorted(raw)
            n = len(ids)
            g = lambda src, k: np.array([src[i].get(k, False) for i in ids], dtype=int)  # noqa: E731
            small = np.array([raw[i]["n_sites"] <= 20 for i in ids])
            row = {"n": n, "oracle": int(g(raw, "oracle").sum()), "phys1": int(g(raw, "phys1").sum()), "phys20": int(g(raw, "phys20").sum())}
            if "fus1" in raw[ids[0]]:
                row["fus1"], row["fus20"] = int(g(raw, "fus1").sum()), int(g(raw, "fus20").sum())
            if ref is not None and "fus1" in ref[ids[0]]:
                r1, r20 = g(ref, "fus1"), g(ref, "fus20")
                row.update({"ref1": int(r1.sum()), "ref20": int(r20.sum()), "ref_oracle": int(g(ref, "oracle").sum()),
                            "ref1_ci": ci_count(r1, rng), "ref20_ci": ci_count(r20, rng),
                            "small_n": int(small.sum()), "small_ref1": int(r1[small].sum()), "small_ref20": int(r20[small].sum()),
                            "small_oracle": int(g(ref, "oracle")[small].sum())})
                per[cond] = (r1, r20)
            table[f"{cond}_lat{lat}"] = row
            f = lambda k: f"{row[k]:>5d}" if k in row else "    -"  # noqa: E731
            print(f"  {LABEL[cond]:26s} {n:4d} {row['oracle']:6d} | {f('phys1'):>6s} {f('phys20'):>7s} | {f('fus1'):>5s} {f('fus20'):>6s} | {f('ref1'):>5s} {f('ref20'):>6s} |        {f('small_ref1'):>5s} {f('small_ref20'):>6s} {f('small_oracle'):>6s}")
        if per:
            print("  paired ladder steps (refined fusion), Δ count [95% CI]:")
            for a, b in STEPS:
                if a in per and b in per:
                    d1 = paired(per[a][0], per[b][0], rng)
                    d20 = paired(per[a][1], per[b][1], rng)
                    print(f"    {LABEL[a]:>16s} → {LABEL[b]:<26s} Top-1 {d1[0]:+d} [{d1[1]:+d}, {d1[2]:+d}]   Top-20 {d20[0]:+d} [{d20[1]:+d}, {d20[2]:+d}]")
                    table[f"step_{a}_to_{b}_lat{lat}"] = {"top1": d1, "top20": d20}
    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=2, default=int))


if __name__ == "__main__":
    main()
