#!/usr/bin/env python3
"""Top-1 / Top-20 under alternative single rankers for an N=100 candidate pool.

Rankers combine the physics rerank score, MLIP energies (CHGNet / MACE) and a
consensus term (size of the StructureMatcher cluster a candidate belongs to):

    rr(x) = 1 / (k + rank(x)),  k = 10
    F_x   = rr(physics) + rr(-E_x)
    +C(w) = ... + w * rr(cluster_size)

Clusters are computed once per pool with the benchmark matcher (same lattice,
so primitive reduction is skipped) and cached next to the pool.

Use --select to print the table for an independent set (tempselect pools), pick
the ranker there, then read the same ranker off CNRS / RRUFF.
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

RPS = {"cnrs": (101, 118), "rruff": (199, 245)}
_SM = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=False, attempt_supercell=False)


def rr(values, k: float = 10.0) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float))
    rank = np.empty(len(values), dtype=float)
    rank[order] = np.arange(len(values))
    return 1.0 / (k + rank)


def load_pool(d: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    paths = (
        sorted(d.glob("test_pxrd_formula_lat1_*_s100.jsonl"))
        or sorted(d.glob("test_*_s100.jsonl"))
        or sorted(d.glob("test_pxrd_formula_lat1_*_s*.jsonl"))
    )
    for p in paths:
        with p.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "candidates" in r:
                        recs.setdefault(r["id"], r)
    return recs


def load_energy(path: Path | None) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    if path is None or not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = {int(s["cand"]): float(s["e_per_atom"]) for s in r["scores"]}
    return out


def to_struct(sd: dict) -> Structure | None:
    try:
        lat = sd["lattice"]
        return Structure(
            Lattice.from_parameters(lat["a"], lat["b"], lat["c"], lat["alpha"], lat["beta"], lat["gamma"]),
            sd["species"], sd["frac_coords"],
        )
    except Exception:
        return None


def cluster_labels(args: tuple[str, list[dict]]) -> tuple[str, list[int]]:
    rid, cands = args
    labels = [-1] * len(cands)
    reps: list[tuple[int, Structure]] = []
    exact: dict[str, int] = {}
    for i, c in enumerate(cands):
        sd = c.get("structure")
        if not sd or not c.get("composition_match"):
            continue
        key = json.dumps(sd, sort_keys=True)
        if key in exact:
            labels[i] = exact[key]
            continue
        s = to_struct(sd)
        if s is None:
            continue
        for lab, rep in reps:
            if len(rep) == len(s) and _SM.fit(rep, s):
                labels[i] = lab
                break
        else:
            labels[i] = len(reps)
            reps.append((len(reps), s))
        exact[key] = labels[i]
    return rid, labels


def cluster_sizes(recs: dict[str, dict], cache: Path, workers: int) -> dict[str, np.ndarray]:
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        with Pool(workers) as pool:
            raw = dict(pool.map(cluster_labels, [(rid, r["candidates"]) for rid, r in recs.items()], chunksize=1))
        cache.write_text(json.dumps(raw))
    out = {}
    for rid, labels in raw.items():
        lab = np.asarray(labels)
        sizes = np.zeros(len(lab), dtype=float)
        for v in set(lab.tolist()):
            if v >= 0:
                sizes[lab == v] = float((lab == v).sum())
        out[rid] = sizes
    return out


def energy_scores(cands: list[dict], e_map: dict[int, float]) -> np.ndarray:
    e = np.full(len(cands), np.nan)
    for k, v in e_map.items():
        if 0 <= k < len(cands):
            e[k] = v
    return np.where(np.isfinite(e), -e, -1e9)


def hit(flags: np.ndarray, score: np.ndarray, k: int) -> bool:
    return bool(flags[np.argsort(-score)[:k]].any())


def build_rankers(weights: list[float]) -> dict[str, callable]:
    R: dict[str, callable] = {
        "physics": lambda p, ec, em, c: p,
        "E_chgnet": lambda p, ec, em, c: ec,
        "E_mace": lambda p, ec, em, c: em,
        "E_avg": lambda p, ec, em, c: rr(ec) + rr(em),
        "F_chgnet": lambda p, ec, em, c: rr(p) + rr(ec),
        "F_mace": lambda p, ec, em, c: rr(p) + rr(em),
        "F_both": lambda p, ec, em, c: rr(p) + rr(ec) + rr(em),
    }
    # fusion hyper-parameters: reciprocal-rank offset k and physics weight
    for k in (5, 20):
        R[f"F_chgnet k={k}"] = lambda p, ec, em, c, k=k: rr(p, k) + rr(ec, k)
    for wp in (0.5, 2.0):
        R[f"F_chgnet physW={wp}"] = lambda p, ec, em, c, wp=wp: wp * rr(p) + rr(ec)
    for w in weights:
        R[f"E_mace+C({w})"] = lambda p, ec, em, c, w=w: rr(em) + w * rr(c)
        R[f"E_chgnet+C({w})"] = lambda p, ec, em, c, w=w: rr(ec) + w * rr(c)
        R[f"E_avg+C({w})"] = lambda p, ec, em, c, w=w: rr(ec) + rr(em) + w * rr(c)
        R[f"F_chgnet+C({w})"] = lambda p, ec, em, c, w=w: rr(p) + rr(ec) + w * rr(c)
        R[f"F_mace+C({w})"] = lambda p, ec, em, c, w=w: rr(p) + rr(em) + w * rr(c)
        R[f"F_both+C({w})"] = lambda p, ec, em, c, w=w: rr(p) + rr(ec) + rr(em) + w * rr(c)
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--chgnet", default="")
    ap.add_argument("--mace", default="")
    ap.add_argument("--rps", choices=["cnrs", "rruff", ""], default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--weights", default="0.25,0.5,1.0")
    ap.add_argument("--match-key", default="structure_match", help="structure_match (refined if pool is refined) or structure_match_raw")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pool_dir = Path(args.pool)
    recs = load_pool(pool_dir)
    ec_map = load_energy(Path(args.chgnet) if args.chgnet else None)
    em_map = load_energy(Path(args.mace) if args.mace else None)
    sizes = cluster_sizes(recs, pool_dir / "clusters.json", args.workers)
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    rankers = build_rankers(weights)

    rows = []
    for rid, r in recs.items():
        cands = r["candidates"]
        flags = np.array([bool(c.get(args.match_key, c.get("structure_match"))) for c in cands])
        phys = np.array([(-1e9 if c.get("rerank_score") is None else float(c["rerank_score"])) for c in cands])
        rows.append((rid, int(r.get("target_n_sites") or 0), flags, phys,
                     energy_scores(cands, ec_map.get(rid, {})), energy_scores(cands, em_map.get(rid, {})), sizes[rid]))
    n = len(rows)
    oracle = sum(int(f.any()) for _, _, f, *_ in rows)
    n_large = sum(1 for _, ns, *_ in rows if ns >= 13)
    print(f"\n=== {args.name or pool_dir.name}  n={n} oracle={oracle} large(n>=13)={n_large} chgnet={bool(ec_map)} mace={bool(em_map)} match_key={args.match_key} ===")
    ref = RPS.get(args.rps) if args.rps else None
    table = {}
    for name, fn in rankers.items():
        if ("chgnet" in name.lower() or "avg" in name or "both" in name) and not ec_map:
            continue
        if ("mace" in name.lower() or "avg" in name or "both" in name) and not em_map:
            continue
        h1 = h20 = 0
        h1_l = h20_l = 0
        for rid, ns, flags, phys, ec, em, cs in rows:
            s = fn(phys, ec, em, cs)
            a, b = hit(flags, s, 1), hit(flags, s, 20)
            h1 += a
            h20 += b
            if ns >= 13:
                h1_l += a
                h20_l += b
        table[name] = {"top1": h1, "top20": h20, "top1_large": h1_l, "top20_large": h20_l}
        extra = f"   vs RPS {ref[0]}/{ref[1]}  Δ={h1-ref[0]:+d}/{h20-ref[1]:+d}" if ref else ""
        print(f"  {name:22s} @1={h1:4d} ({100*h1/n:5.1f}%)  @20={h20:4d} ({100*h20/n:5.1f}%)   large {h1_l}/{h20_l}{extra}")
    if args.out:
        Path(args.out).write_text(json.dumps({"n": n, "oracle": oracle, "rankers": table}, indent=2))


if __name__ == "__main__":
    main()
