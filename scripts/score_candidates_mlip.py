#!/usr/bin/env python3
"""Score stored generation candidates with CHGNet or MACE-MPA-0.

Writes one JSONL row per target:
  {"id": ..., "scores": [{"cand": i, "e_per_atom": ...}, ...]}
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


def structure_of(sd: dict):
    from pymatgen.core import Lattice, Structure

    L = sd["lattice"]
    return Structure(
        Lattice.from_parameters(L["a"], L["b"], L["c"], L["alpha"], L["beta"], L["gamma"]),
        sd["species"],
        sd["frac_coords"],
    )


def load_pool(eval_dir: Path, glob_pat: str, cond: str) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    paths = sorted(eval_dir.glob(glob_pat))
    if not paths:
        paths = sorted(eval_dir.glob(f"test_{cond}_lat1_*_s100.jsonl"))
    if not paths:
        paths = sorted(eval_dir.glob(f"test_{cond}_lat1_*_s*.jsonl"))
    for path in paths:
        if path.name.endswith("_summary.json"):
            continue
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                pool.setdefault(row["id"], row)
    return pool


def score_chgnet(structs: list, batch_size: int) -> list[float]:
    from chgnet.model import CHGNet

    if not hasattr(score_chgnet, "model"):
        score_chgnet.model = CHGNet.load()
    model = score_chgnet.model
    try:
        preds = model.predict_structure(structs, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001
        import torch

        oom = "out of memory" in str(exc).lower() or "OutOfMemory" in type(exc).__name__
        if not oom:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        preds = model.predict_structure(structs, batch_size=1)
    out = []
    for pr in preds:
        e = float(pr["e"]) if pr is not None else float("nan")
        out.append(e)
    return out


def score_mace(structs: list, batch_size: int) -> list[float]:
    from mace.calculators import mace_mp
    from pymatgen.io.ase import AseAtomsAdaptor

    if not hasattr(score_mace, "calc"):
        score_mace.calc = mace_mp(
            model="medium-mpa-0",
            device="cuda",
            default_dtype="float32",
        )
        score_mace.adaptor = AseAtomsAdaptor()
    calc = score_mace.calc
    adaptor = score_mace.adaptor
    energies: list[float] = []
    for start in range(0, len(structs), batch_size):
        chunk = structs[start : start + batch_size]
        atoms_list = [adaptor.get_atoms(s) for s in chunk]
        try:
            for atoms in atoms_list:
                atoms.calc = calc
                e_tot = float(atoms.get_potential_energy())
                energies.append(e_tot / max(1, len(atoms)))
        except Exception:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for atoms in atoms_list:
                try:
                    atoms.calc = calc
                    e_tot = float(atoms.get_potential_energy())
                    energies.append(e_tot / max(1, len(atoms)))
                except Exception:
                    energies.append(float("nan"))
    return energies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["chgnet", "mace"], default="mace")
    ap.add_argument("--cond", default="pxrd_formula")
    ap.add_argument("--glob", default="")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    glob_pat = args.glob or f"test_{args.cond}_lat1_*_s100.jsonl"
    pool = load_pool(eval_dir, glob_pat, args.cond)
    ids = sorted(pool)
    if args.limit:
        ids = ids[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    mode = "w"
    if args.resume and out.exists():
        with out.open() as handle:
            for line in handle:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        mode = "a"
        ids = [rid for rid in ids if rid not in done_ids]
        print(f"resume skip={len(done_ids)} remaining={len(ids)}", file=sys.stderr, flush=True)

    score_fn = score_mace if args.backend == "mace" else score_chgnet
    done = 0
    with out.open(mode) as handle:
        for rid in ids:
            cands = pool[rid].get("candidates") or []
            idx = [i for i, c in enumerate(cands) if c.get("structure")]
            if not idx:
                continue
            try:
                structs = [structure_of(cands[i]["structure"]) for i in idx]
            except Exception:
                continue
            try:
                energies = score_fn(structs, args.batch_size)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {rid}: {type(exc).__name__} {exc}", file=sys.stderr, flush=True)
                continue
            rows = []
            for i, e in zip(idx, energies):
                if not np.isfinite(e):
                    continue
                rows.append({"cand": i, "e_per_atom": round(float(e), 5)})
            if rows:
                handle.write(json.dumps({"id": rid, "scores": rows}) + "\n")
                handle.flush()
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{len(ids)}", file=sys.stderr, flush=True)
    print(f"wrote {out}  targets={done} backend={args.backend}", file=sys.stderr)


if __name__ == "__main__":
    main()
