#!/usr/bin/env python3
"""Offline reranking lab over frozen candidate pools.

Generation is the expensive step and its output is already stored, so scoring
ideas can be tried here without a GPU sampler.

Only information the model was given may become a feature: the measured
pattern, the formula, the unit cell, and the candidate structure itself. The
ground-truth structure and anything simulated from it (including the reference
neutron pattern) are used to report accuracy, never to score.

Two properties the current reranker throws away:

* With the cell known, every candidate for a target puts its reflections at the
  same 2-theta, so peak positions carry no discriminative information and the
  broadened-profile cosine is mostly measuring agreement that is common to all
  candidates. Comparing reflection intensities directly is sharper.
* A structure the sampler drew many times out of 100 is far more likely to be
  right than a one-off. Today multiplicity only pushes duplicates down the list.

Weights are fitted on pools from other datasets and applied unchanged to the
held-out ones, so reported gains are not tuned on the set they are measured on.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crystal_language.coord_refine import FixedLatticePXRD, RefineConfig  # noqa: E402
from crystal_language.rerank import structure_dedup_key  # noqa: E402

FEATURES = [
    "cos_refl",
    "cos_log",
    "cos_sqrt",
    "rwp_neg",
    "log_mult",
    "sg_num",
    "sg_is_high",
    "charge_ok",
    "stored",
]


def load_pool(eval_dir: Path, cond: str = "pxrd_formula", lat: int = 1) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for p in sorted(eval_dir.glob(f"test_{cond}_lat{lat}_*_s100.jsonl")):
        for line in p.open():
            r = json.loads(line)
            recs.setdefault(r["id"], r)
    return recs


def load_native(path: Path) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def target_matrix(rec: dict, nat: dict, cfg: RefineConfig):
    """Return (flags, feature_matrix, feature_names) for one target."""
    cands = [c for c in (rec.get("candidates") or []) if c.get("structure")]
    if len(cands) < 2:
        return None
    lat0 = cands[0]["structure"]["lattice"]
    species = list(cands[0]["structure"]["species"])
    keep = [
        c
        for c in cands
        if list(c["structure"]["species"]) == species and c["structure"]["lattice"] == lat0
    ]
    if len(keep) < 2:
        return None

    from pymatgen.core import Lattice

    lattice = Lattice.from_parameters(
        lat0["a"], lat0["b"], lat0["c"], lat0["alpha"], lat0["beta"], lat0["gamma"]
    )
    try:
        op = FixedLatticePXRD(lattice, species, cfg)
        obs = op.observed_profile(nat["conditions"]["pxrd_x"], nat["conditions"]["pxrd_y"])
    except Exception:
        return None

    frac = torch.tensor(
        np.stack([np.mod(np.asarray(c["structure"]["frac_coords"], dtype=np.float64), 1.0) for c in keep]),
        dtype=torch.float32,
        device=op.device,
    )
    with torch.no_grad():
        prof = op.profile(frac)
        o = obs[None, :]

        def cos(a, b):
            return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)).cpu().numpy()

        f_cos = cos(prof, o)
        f_sqrt = cos(prof.sqrt(), o.sqrt())
        f_log = cos(torch.log1p(10 * prof), torch.log1p(10 * o))
        w = 1.0 / o.clamp_min(1e-3)
        scale = (w * prof * o).sum(-1) / (w * prof * prof).sum(-1).clamp_min(1e-12)
        num = (w * (o - scale[:, None] * prof) ** 2).sum(-1)
        den = (w * o**2).sum(-1).clamp_min(1e-12)
        f_rwp = -torch.sqrt(num / den).cpu().numpy()

    counts = Counter(structure_dedup_key(c["structure"]) for c in keep)
    f_mult = np.log1p([float(counts[structure_dedup_key(c["structure"])]) for c in keep])
    sg = np.array([float(c.get("space_group") or 1) for c in keep])
    f_sg_high = (sg >= 195).astype(float)
    charge = np.array([float(bool((c.get("rerank") or {}).get("charge_ok"))) for c in keep])
    stored = np.array([float(c.get("rerank_score") or -1e9) for c in keep])

    cols = {
        "cos_refl": f_cos,
        "cos_log": f_log,
        "cos_sqrt": f_sqrt,
        "rwp_neg": f_rwp,
        "log_mult": f_mult,
        "sg_num": sg / 230.0,
        "sg_is_high": f_sg_high,
        "charge_ok": charge,
        "stored": stored,
    }
    X = np.stack([cols[f] for f in FEATURES], axis=1)
    flags = np.array([bool(c.get("structure_match")) for c in keep])
    return flags, X


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / np.where(sd < 1e-9, 1.0, sd)


def collect(eval_dir: str, native: str, cfg: RefineConfig):
    recs = load_pool(Path(eval_dir))
    nat = load_native(Path(native))
    out = []
    for rid in sorted(recs):
        if rid not in nat:
            continue
        got = target_matrix(recs[rid], nat[rid], cfg)
        if got is None:
            continue
        flags, X = got
        out.append((rid, flags, X, zscore(X)))
    return out


def evaluate(targets, weights: np.ndarray | None, feature: str | None = None) -> dict:
    n = len(targets)
    oracle = t1 = t5 = 0
    for _, flags, X, Z in targets:
        if not flags.any():
            continue
        oracle += 1
        s = Z @ weights if weights is not None else X[:, FEATURES.index(feature)]
        order = np.argsort(-s)
        t1 += int(flags[order[0]])
        t5 += int(flags[order[:5]].any())
    return {"n": n, "oracle": oracle, "top1": t1, "top5": t5}


def fit_logistic(targets, l2: float = 1.0, iters: int = 400, lr: float = 0.2) -> np.ndarray:
    Z = np.concatenate([t[3] for t in targets if t[1].any()], axis=0)
    y = np.concatenate([t[1].astype(float) for t in targets if t[1].any()], axis=0)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= lr * grad
    return w


def report(label: str, res: dict) -> None:
    n, orc, t1, t5 = res["n"], res["oracle"], res["top1"], res["top5"]
    share = 100 * t1 / orc if orc else 0.0
    print(
        f"  {label:24s} top-1 {t1:4d}/{n} ({100*t1/max(n,1):5.1f}%)   "
        f"top-5 {t5:4d} ({100*t5/max(n,1):5.1f}%)   兑现 oracle 的 {share:5.1f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[], help="eval_dir:native pairs used to fit weights")
    ap.add_argument("--test", nargs="+", required=True, help="eval_dir:native:label triples")
    ap.add_argument("--sigma", type=float, default=0.20)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = RefineConfig(sigma=args.sigma)

    weights = None
    if args.train:
        train = []
        for spec in args.train:
            ed, nv = spec.split(":")[:2]
            got = collect(ed, nv, cfg)
            print(f"train pool {Path(ed).name}: {len(got)} targets", file=sys.stderr)
            train += got
        weights = fit_logistic(train, l2=args.l2)
        print("\nfitted weights (z-scored features):")
        for f, wv in sorted(zip(FEATURES, weights), key=lambda kv: -abs(kv[1])):
            print(f"  {f:>12s} {wv:+.3f}")

    results = {}
    for spec in args.test:
        parts = spec.split(":")
        ed, nv = parts[0], parts[1]
        label = parts[2] if len(parts) > 2 else Path(ed).name
        targets = collect(ed, nv, cfg)
        orc = sum(1 for t in targets if t[1].any())
        print(f"\n=== {label} ===  targets {len(targets)}  oracle {orc} ({100*orc/max(len(targets),1):.1f}%)")
        for f in ["stored", "cos_refl", "log_mult"]:
            report(f"single: {f}", evaluate(targets, None, f))
        if weights is not None:
            res = evaluate(targets, weights)
            report("fitted combination", res)
            results[label] = res
    if args.out and results:
        Path(args.out).write_text(
            json.dumps({"weights": dict(zip(FEATURES, weights.tolist())), "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
