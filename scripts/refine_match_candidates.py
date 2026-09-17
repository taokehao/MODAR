#!/usr/bin/env python3
"""Refine generated candidates before structure matching (RealPXRD-Solver protocol parity).

RPS passes its 20 candidates through a scripted Rietveld refinement of atomic positions before
comparing to the reference CIF. Our candidates were matched as generated. This script applies a
frozen-lattice MLIP relaxation (CHGNet by default) to the unmatched candidates of each target,
re-runs the benchmark StructureMatcher on the relaxed geometry, and writes a new pool in the
same jsonl format:

  structure_match      = raw match OR refined match   (used by downstream analysis)
  structure_match_raw  = original flag
  refined              = {"matched", "rms", "max_dist", "e_relaxed", "steps"} for relaxed reps

Candidate order and indices are unchanged, so existing energy files (indexed by candidate
position) remain valid. Ranking scores are not touched: refinement only decides the match.

Unmatched candidates are deduplicated (exact dict) and relaxed in order of physics rerank score,
up to --max-relax per target (RPS refines all of its 20). With --cluster they are first grouped
with the benchmark matcher and one representative per cluster is relaxed (slow for large cells);
a rescued representative rescues its whole group either way.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

STRICT = dict(ltol=0.3, stol=0.5, angle_tol=10)


def to_struct(sd: dict) -> Structure | None:
    try:
        lat = sd["lattice"]
        L = Lattice.from_parameters(lat["a"], lat["b"], lat["c"], lat["alpha"], lat["beta"], lat["gamma"])
        return Structure(L, sd["species"], sd["frac_coords"])
    except Exception:
        return None


def struct_to_dict(s: Structure) -> dict:
    lat = s.lattice
    return {
        "lattice": {"a": lat.a, "b": lat.b, "c": lat.c, "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma},
        "species": [str(x.specie) for x in s],
        "frac_coords": [[float(v) for v in fc] for fc in s.frac_coords],
    }


def load_refs(native: Path) -> dict[str, Structure]:
    """References from a native jsonl (explicit coordinates) or from a crystal_language_v2
    test.jsonl (tokenised target, decoded exactly as evaluate_generation does)."""
    refs: dict[str, Structure] = {}
    with native.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("frac_coords") is None and r.get("target"):
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from crystal_language.decode import parse_crystal_tokens, to_structure

                try:
                    refs[r["id"]] = to_structure(parse_crystal_tokens(r["target"].split()))
                except Exception:  # noqa: BLE001
                    pass
                continue
            if r.get("lattice_matrix") is not None:
                L = Lattice(np.asarray(r["lattice_matrix"], dtype=float).reshape(3, 3))
            else:
                lp = r["lattice_parameters"]
                L = Lattice.from_parameters(*[float(lp[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")]) if isinstance(lp, dict) else Lattice.from_parameters(*map(float, lp))
            refs[r["id"]] = Structure(L, r["atom_types"], r["frac_coords"])
    return refs


def cluster_unmatched(cands: list[dict], same_lattice_sm: StructureMatcher) -> list[list[int]]:
    """Greedy tolerance clustering of unmatched, composition-correct candidates."""
    exact: dict[str, int] = {}
    reps: list[tuple[Structure, list[int]]] = []
    for i, c in enumerate(cands):
        sd = c.get("structure")
        if not sd or not c.get("composition_match") or c.get("structure_match"):
            continue
        key = json.dumps(sd, sort_keys=True)
        if key in exact:
            reps[exact[key]][1].append(i)
            continue
        s = to_struct(sd)
        if s is None:
            continue
        for k, (rep, members) in enumerate(reps):
            if len(rep) == len(s) and same_lattice_sm.fit(rep, s):
                members.append(i)
                exact[key] = k
                break
        else:
            exact[key] = len(reps)
            reps.append((s, [i]))
    return [members for _, members in reps]


def dedupe_unmatched(cands: list[dict]) -> list[list[int]]:
    """Exact-duplicate groups of unmatched, composition-correct candidates, best physics score first."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(cands):
        sd = c.get("structure")
        if not sd or not c.get("composition_match") or c.get("structure_match"):
            continue
        groups.setdefault(json.dumps(sd, sort_keys=True), []).append(i)
    out = list(groups.values())
    out.sort(key=lambda m: -(cands[m[0]].get("rerank_score") if cands[m[0]].get("rerank_score") is not None else -1e9))
    return out


def _rr(values: np.ndarray, k: float = 10.0) -> np.ndarray:
    order = np.argsort(-values)
    rank = np.empty(len(values), dtype=float)
    rank[order] = np.arange(len(values))
    return 1.0 / (k + rank)


def fusion_top_unmatched(cands: list[dict], e_map: dict[int, float], top: int, include_matched: bool = False) -> list[list[int]]:
    """Candidates inside the fusion top list, in fusion order (one group each).

    By default only unmatched ones (the match verdict is all that refinement can change);
    ``include_matched`` also relaxes already-matched candidates so that relaxation-derived
    features (displacement, relaxed energy) exist for every reported structure without
    depending on the label."""
    phys = np.array([(-1e9 if c.get("rerank_score") is None else float(c["rerank_score"])) for c in cands])
    e = np.full(len(cands), np.nan)
    for k, v in e_map.items():
        if 0 <= k < len(cands):
            e[k] = v
    es = np.where(np.isfinite(e), -e, -1e9)
    fused = _rr(phys) + _rr(es)
    out = []
    for i in np.argsort(-fused)[:top]:
        c = cands[int(i)]
        if c.get("structure") and c.get("composition_match") and (include_matched or not c.get("structure_match")):
            out.append([int(i)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--native", required=True, help="reference native/test.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backend", choices=["chgnet", "mace"], default="chgnet")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-relax", type=int, default=40, help="max candidates (or cluster reps) relaxed per target")
    ap.add_argument("--cluster", action="store_true", help="cluster unmatched candidates first (slow); default: exact dedupe, top by rerank_score")
    ap.add_argument("--glob", default="test_pxrd_formula_lat1_*_s100.jsonl")
    ap.add_argument("--only-ids", default="", help="comma-separated ids to restrict to (debug)")
    ap.add_argument(
        "--rank-by", choices=["physics", "fusion"], default="physics",
        help="fusion: relax the unmatched candidates inside the fusion (physics+energy) top --top list, i.e. the "
             "structures actually reported, mirroring RPS refining its 20 outputs; needs --energy",
    )
    ap.add_argument("--energy", default="", help="energy jsonl (per-atom MLIP energies) for --rank-by fusion")
    ap.add_argument("--top", type=int, default=20, help="list length for --rank-by fusion")
    ap.add_argument("--include-matched", action="store_true", help="with --rank-by fusion: relax matched candidates too")
    args = ap.parse_args()

    eval_dir, out_dir = Path(args.eval_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_refs(Path(args.native))
    energies: dict[str, dict[int, float]] = {}
    if args.rank_by == "fusion":
        if not args.energy:
            raise SystemExit("--rank-by fusion needs --energy")
        with Path(args.energy).open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    energies[r["id"]] = {int(s["cand"]): float(s["e_per_atom"]) for s in r["scores"]}
    strict = StructureMatcher(primitive_cell=True, **STRICT)
    same_lat = StructureMatcher(primitive_cell=False, attempt_supercell=False, **STRICT)
    only = set(x for x in args.only_ids.split(",") if x)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.backend == "chgnet":
        from chgnet.model import StructOptimizer

        relaxer = StructOptimizer(use_device=device)

        def relax(s: Structure) -> tuple[Structure, float, int]:
            out = relaxer.relax(s, fmax=args.fmax, steps=args.steps, relax_cell=False, verbose=False)
            fs = out["final_structure"]
            e = float(out["trajectory"].energies[-1]) / len(fs)
            return fs, e, len(out["trajectory"].energies)
    else:
        from ase.filters import FrechetCellFilter  # noqa: F401  (import check only)
        from ase.optimize import FIRE
        from mace.calculators import mace_mp
        from pymatgen.io.ase import AseAtomsAdaptor

        calc = mace_mp(model="medium-mpa-0", device=device, default_dtype="float32")

        def relax(s: Structure) -> tuple[Structure, float, int]:
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.calc = calc
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=args.fmax, steps=args.steps)
            fs = AseAtomsAdaptor.get_structure(atoms)
            return fs, float(atoms.get_potential_energy()) / len(atoms), opt.nsteps

    paths = sorted(eval_dir.glob(args.glob))
    t_start = time.time()
    n_targets = n_rescued = n_relaxed = 0
    for path in paths:
        out_path = out_dir / path.name
        if out_path.exists():
            print(f"skip existing {out_path.name}", flush=True)
            continue
        rows_out = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row["id"]
                cands = row.get("candidates") or []
                for c in cands:
                    # keep the original flag when re-running on an already refined pool
                    c.setdefault("structure_match_raw", bool(c.get("structure_match")))
                if only and rid not in only or rid not in refs:
                    rows_out.append(row)
                    continue
                n_targets += 1
                ref = refs[rid]
                if args.rank_by == "fusion":
                    clusters = fusion_top_unmatched(cands, energies.get(rid, {}), args.top, args.include_matched)
                elif args.cluster:
                    clusters = cluster_unmatched(cands, same_lat)
                    clusters.sort(key=lambda m: (-len(m), -(cands[m[0]].get("rerank_score") or -1e9)))
                else:
                    clusters = dedupe_unmatched(cands)
                # candidates relaxed in an earlier pass carry their verdict already
                clusters = [m for m in clusters if "refined" not in cands[m[0]]]
                rescued_here = 0
                for members in clusters[: args.max_relax]:
                    rep = cands[members[0]]
                    s0 = to_struct(rep["structure"])
                    if s0 is None or len(s0) != len(ref):
                        continue
                    try:
                        s1, e_rel, steps = relax(s0)
                    except Exception as exc:  # noqa: BLE001
                        rep["refined"] = {"error": str(exc)[:120]}
                        continue
                    n_relaxed += 1
                    matched = bool(strict.fit(ref, s1))
                    d = strict.get_rms_dist(ref, s1) if matched else None
                    info = {
                        "matched": matched,
                        "rms": (float(d[0]) if d else None),
                        "max_dist": (float(d[1]) if d else None),
                        "e_relaxed": e_rel,
                        "steps": steps,
                        "cluster_size": len(members),
                    }
                    rep["refined"] = info
                    rep["refined_structure"] = struct_to_dict(s1)
                    if matched:
                        rescued_here += 1
                        for i in members:
                            cands[i]["structure_match"] = True
                            cands[i]["structure_match_refined"] = True
                if rescued_here:
                    n_rescued += 1
                row["refined_rescued_clusters"] = rescued_here
                row["oracle_raw"] = any(c.get("structure_match_raw") for c in cands)
                row["oracle_refined"] = any(c.get("structure_match") for c in cands)
                rows_out.append(row)
                print(
                    f"{rid} n={row.get('target_n_sites')} clusters={len(clusters)} relaxed={min(len(clusters), args.max_relax)}"
                    f" raw={row['oracle_raw']} refined={row['oracle_refined']} elapsed={time.time()-t_start:.0f}s",
                    flush=True,
                )
        with out_path.open("w") as f:
            for row in rows_out:
                f.write(json.dumps(row) + "\n")
    print(f"done targets={n_targets} relaxed={n_relaxed} targets_with_rescue={n_rescued} in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
