#!/usr/bin/env python3
"""Refine-then-rerank: does information from the relaxed top-K change Top-1?

For each target the fusion top-K (physics + CHGNet single-point energy) is the
reported list; those candidates were relaxed at fixed lattice by
``refine_match_candidates.py`` and carry ``refined_structure`` / ``refined``.
Here we compute, for the relaxed candidates, the relaxed energy, the physics
rerank score of the relaxed geometry (pattern agreement after refinement), the
displacement during relaxation, and the population of the relaxed basin, and
test rankers built from them. Top-20 is unaffected (same set); only the order
inside the reported list changes, so this is a Top-1 lever.

Run on the independent tempselect pools first to choose a ranker.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crystal_language.rerank import SIGMA, rerank_candidates  # noqa: E402

RPS = {"cnrs": (101, 118), "rruff": (199, 245)}
_SM = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=False, attempt_supercell=False)


def rr(values, k: float = 10.0) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    order = np.argsort(-v)
    rank = np.empty(len(v), dtype=float)
    rank[order] = np.arange(len(v))
    return 1.0 / (k + rank)


def load_pool(d: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for p in sorted(d.glob("test_pxrd_formula_lat1_*_s100.jsonl")):
        with p.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "candidates" in r:
                        recs.setdefault(r["id"], r)
    return recs


def load_energy(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = {int(s["cand"]): float(s["e_per_atom"]) for s in r["scores"]}
    return out


def load_spectra(path: Path) -> dict[str, list[tuple[float, float]]]:
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sp = r.get("spectra") or {}
                out[r["id"]] = list(zip(sp.get("pxrd_x") or [], sp.get("pxrd_y") or []))
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


def displacement(a: dict, b: dict) -> float:
    """Mean per-atom displacement (angstrom) between two structure dicts on the same lattice."""
    s = to_struct(a)
    if s is None:
        return float("nan")
    fa = np.asarray(a["frac_coords"], dtype=float)
    fb = np.asarray(b["frac_coords"], dtype=float)
    d = fb - fa
    d -= np.round(d)
    cart = d @ s.lattice.matrix
    return float(np.linalg.norm(cart, axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="refined pool (with refined_structure)")
    ap.add_argument("--chgnet", required=True)
    ap.add_argument("--records", required=True, help="crystal_language_v2/test.jsonl with spectra")
    ap.add_argument("--rps", choices=["cnrs", "rruff", ""], default="")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    recs = load_pool(Path(args.pool))
    E = load_energy(Path(args.chgnet))
    spectra = load_spectra(Path(args.records))
    n = len(recs)

    # per-target feature tables restricted to the fusion top-K
    tables = []
    for rid, r in recs.items():
        cands = r["candidates"]
        flags = np.array([bool(c.get("structure_match")) for c in cands])
        phys = np.array([(-1e9 if c.get("rerank_score") is None else float(c["rerank_score"])) for c in cands])
        e = np.full(len(cands), np.nan)
        for k, v in E.get(rid, {}).items():
            if 0 <= k < len(cands):
                e[k] = v
        es = np.where(np.isfinite(e), -e, -1e9)
        fused = rr(phys) + rr(es)
        top = [int(i) for i in np.argsort(-fused)[: args.top]]
        # relaxed geometry for each top candidate (fall back to the raw one)
        payloads = []
        e_rel = np.full(len(top), np.nan)
        disp = np.full(len(top), np.nan)
        for j, i in enumerate(top):
            c = cands[i]
            sd = c.get("refined_structure") or c.get("structure")
            payloads.append({"lattice": sd["lattice"], "species": sd["species"], "frac_coords": sd["frac_coords"]} if sd else None)
            info = c.get("refined") or {}
            if info.get("e_relaxed") is not None:
                e_rel[j] = float(info["e_relaxed"])
                disp[j] = displacement(c["structure"], c["refined_structure"])
            elif np.isfinite(e[i]):
                # matched candidates were not relaxed: treat the single-point value as the basin energy
                e_rel[j] = float(e[i])
                disp[j] = 0.0
        ref_pxrd = spectra.get(rid)
        phys_rel = np.full(len(top), -1e9)
        if ref_pxrd:
            out = rerank_candidates(payloads, ref_pxrd, None, executor=None, match_sigma=SIGMA, shift_max=0.0)
            for j, s in enumerate(out.scores):
                if s is not None:
                    phys_rel[j] = float(s)
        # basin population: cluster relaxed geometries among the top-K
        pop = np.ones(len(top))
        structs = [to_struct(p) if p else None for p in payloads]
        labels = [-1] * len(top)
        reps: list[tuple[int, Structure]] = []
        for j, s in enumerate(structs):
            if s is None:
                continue
            for lab, rep in reps:
                if len(rep) == len(s) and _SM.fit(rep, s):
                    labels[j] = lab
                    break
            else:
                labels[j] = len(reps)
                reps.append((len(reps), s))
        for lab in set(labels):
            if lab >= 0:
                idx = [j for j, l in enumerate(labels) if l == lab]
                for j in idx:
                    pop[j] = len(idx)
        tables.append({
            "id": rid, "n_sites": int(r.get("target_n_sites") or 0), "top": top, "flags": flags[top],
            "fused": fused[top],
            "phys": phys[top], "es": es[top], "e_rel": np.where(np.isfinite(e_rel), -e_rel, -1e9),
            "phys_rel": phys_rel, "disp": np.where(np.isfinite(disp), -disp, -1e9), "pop": pop,
        })

    # "F" is the full-pool fusion score (the deployed ranker); add-on terms are
    # reciprocal ranks computed inside the reported top-K.
    rankers = {
        "F_chgnet (baseline)": lambda t: t["fused"],
        "F + relaxedE(1.0)": lambda t: t["fused"] + rr(t["e_rel"]),
        "F + relaxedPhys(1.0)": lambda t: t["fused"] + rr(t["phys_rel"]),
        "F + pop(0.25)": lambda t: t["fused"] + 0.25 * rr(t["pop"]),
        "F + pop(0.5)": lambda t: t["fused"] + 0.5 * rr(t["pop"]),
        "F + pop(1.0)": lambda t: t["fused"] + 1.0 * rr(t["pop"]),
        "F + smallDisp(0.25)": lambda t: t["fused"] + 0.25 * rr(t["disp"]),
        "F + smallDisp(0.5)": lambda t: t["fused"] + 0.5 * rr(t["disp"]),
        "F + smallDisp(1.0)": lambda t: t["fused"] + 1.0 * rr(t["disp"]),
        "F + pop(0.5) + smallDisp(0.5)": lambda t: t["fused"] + 0.5 * rr(t["pop"]) + 0.5 * rr(t["disp"]),
        "topK re-fusion relaxedPhys+relaxedE": lambda t: rr(t["phys_rel"]) + rr(t["e_rel"]),
        "relaxedE only": lambda t: t["e_rel"],
        "relaxedPhys only": lambda t: t["phys_rel"],
        "pop then F": lambda t: t["pop"] + 1e-3 * t["fused"],
    }
    oracle_top = sum(int(t["flags"].any()) for t in tables)
    ref = RPS.get(args.rps) if args.rps else None
    print(f"\n=== {args.name or args.pool}  n={n}  top{args.top} contains a match for {oracle_top} targets (= Top-{args.top} for every ranker) ===")
    for name, fn in rankers.items():
        h1 = 0
        for t in tables:
            s = np.asarray(fn(t), dtype=float)
            h1 += int(t["flags"][int(np.argmax(s))])
        extra = f"   vs RPS T1 {ref[0]}  Δ={h1-ref[0]:+d}" if ref else ""
        print(f"  {name:26s} @1={h1:4d} ({100*h1/n:5.1f}%){extra}")


if __name__ == "__main__":
    main()
