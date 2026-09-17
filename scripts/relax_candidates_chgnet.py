#!/usr/bin/env python3
"""CHGNet ionic relaxation of stored candidates (cell frozen).

Known-lattice protocol: only fractional coordinates move. Relaxes the fusion
(or physics) top-k per target and writes relaxed energy per atom plus a
post-relax StructureMatcher flag against the ground-truth crystal.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rr(values: np.ndarray, k: float = 10.0) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float))
    rank = np.empty(len(values), dtype=float)
    rank[order] = np.arange(len(values))
    return 1.0 / (k + rank)


def structure_of(sd: dict):
    from pymatgen.core import Lattice, Structure

    L = sd["lattice"]
    return Structure(
        Lattice.from_parameters(L["a"], L["b"], L["c"], L["alpha"], L["beta"], L["gamma"]),
        sd["species"],
        sd["frac_coords"],
    )


def load_eval(eval_dir: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for path in sorted(eval_dir.glob("test_pxrd_formula_lat1_*_s100.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                recs.setdefault(row["id"], row)
    return recs


def load_chgnet(path: Path | None) -> dict[str, dict[int, float]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[int, float]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[row["id"]] = {int(s["cand"]): float(s["e_per_atom"]) for s in row["scores"]}
    return out


def load_targets(data_dir: Path) -> dict[str, object]:
    from crystal_language.decode import parse_crystal_tokens, to_structure

    out = {}
    with (data_dir / "test.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            parsed = parse_crystal_tokens(row["target"].split())
            out[row["id"]] = to_structure(parsed)
    return out


def pick_indices(row: dict, chg: dict[int, float], top_k: int) -> list[int]:
    cands = row.get("candidates") or []
    n = len(cands)
    phys = np.array(
        [(-1e9 if c.get("rerank_score") is None else float(c["rerank_score"])) for c in cands],
        dtype=float,
    )
    energy = np.full(n, np.nan)
    for i, e in chg.items():
        if 0 <= i < n:
            energy[i] = e
    e_score = np.where(np.isfinite(energy), -energy, -1e9)
    fused = rr(phys) + rr(e_score)
    have = [i for i, c in enumerate(cands) if c.get("structure")]
    if not have:
        return []
    order = [i for i in np.argsort(-fused) if i in set(have)]
    return [int(i) for i in order[:top_k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--chgnet", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--fmax", type=float, default=0.2)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.analysis.structure_matcher import StructureMatcher

    recs = load_eval(Path(args.eval_dir))
    chg_all = load_chgnet(Path(args.chgnet) if args.chgnet else None)
    ids = sorted(recs)
    if args.limit:
        ids = ids[: args.limit]
    ids = [rid for i, rid in enumerate(ids) if i % args.num_shards == args.shard_index]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    mode = "w"
    if args.resume and out.exists():
        with out.open() as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["id"])
        mode = "a"
        ids = [rid for rid in ids if rid not in done]
        print(f"resume skip={len(done)} remaining={len(ids)}", flush=True)

    print(f"CHGNet load  shard={args.shard_index}/{args.num_shards}  targets={len(ids)}", flush=True)
    model = CHGNet.load()
    relaxer = StructOptimizer(model=model)
    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True)
    targets = load_targets(Path(args.data_dir))

    n_done = 0
    with out.open(mode) as fh:
        for rid in ids:
            row = recs[rid]
            cands = row.get("candidates") or []
            idxs = pick_indices(row, chg_all.get(rid, {}), args.top_k)
            gt = targets.get(rid)
            scores = []
            for i in idxs:
                try:
                    struct = structure_of(cands[i]["structure"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {rid}[{i}] build {type(exc).__name__}", flush=True)
                    continue
                n_sites = max(1, len(struct))
                try:
                    result = relaxer.relax(
                        struct,
                        fmax=args.fmax,
                        steps=args.steps,
                        relax_cell=False,
                        verbose=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {rid}[{i}] relax {type(exc).__name__}: {exc}", flush=True)
                    continue
                traj = result["trajectory"]
                e_tot = float(traj.energies[-1]) if traj.energies else float("nan")
                e_pa = e_tot / n_sites
                final = result["final_structure"]
                match_after = None
                if gt is not None:
                    try:
                        match_after = bool(matcher.fit(final, gt))
                    except Exception:
                        match_after = False
                scores.append(
                    {
                        "cand": int(i),
                        "e_relax": round(float(e_pa), 5),
                        "n_steps": int(len(traj.energies)),
                        "match_after": bool(match_after) if match_after is not None else None,
                    }
                )
            fh.write(json.dumps({"id": rid, "scores": scores}) + "\n")
            fh.flush()
            n_done += 1
            print(f"  {n_done}/{len(ids)} {rid} relaxed={len(scores)}", flush=True)
    print(f"wrote {out}  targets={n_done}", flush=True)


if __name__ == "__main__":
    main()
