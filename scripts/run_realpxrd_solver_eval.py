#!/usr/bin/env python3
"""Evaluate a RealPXRD-Solver checkpoint on a gzipped LMDB test set.

Replicates eval.sh:
  Top-1  = one unranked sample
  Top-20 = oracle over 20 unranked samples (min RMS)
Matcher: StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10) with default
primitive_cell=False. SMACT+construction validity is applied as in compute.py.

Does not instantiate their train/valid LMDBs (those paths are on the authors'
cluster). Lattice-conditioned checkpoints are detected via cost_lattice~0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[1]
RPS = ROOT / "third_party" / "RealPXRD-Solver"
sys.path.insert(0, str(RPS))
sys.path.insert(0, str(RPS / "scripts"))

from app.data.dataset import CrystDataset_pxrd  # noqa: E402
from app.model.flow import CSPFlow  # noqa: E402
from eval_utils import get_crystals_list, get_crystals_true_list  # noqa: E402
from torch_geometric.data import DataLoader  # noqa: E402


def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.clamp(
            torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1)
            / (lengths[..., j] * lengths[..., k]),
            -1.0,
            1.0,
        )
    angles = torch.arccos(angles) * 180.0 / torch.pi
    return lengths, angles


def sample_loader(loader, model, num_evals: int, max_batches: int | None):
    frac_coords, num_atoms, atom_types, lattices, input_data_list = [], [], [], [], []
    for bidx, batch in enumerate(loader):
        batch = batch.to(model.device)
        batch_frac, batch_natoms, batch_types, batch_lat = [], [], [], []
        for _ in range(num_evals):
            outputs = model.sample(batch, step_lr=5, infer_timesteps=200)
            batch_frac.append(outputs["frac_coords"].detach().cpu())
            batch_natoms.append(outputs["num_atoms"].detach().cpu())
            batch_types.append(outputs["atom_types"].detach().cpu())
            batch_lat.append(outputs["lattices"].detach().cpu())
        frac_coords.append(torch.stack(batch_frac, dim=0))
        num_atoms.append(torch.stack(batch_natoms, dim=0))
        atom_types.append(torch.stack(batch_types, dim=0))
        lattices.append(torch.stack(batch_lat, dim=0))
        input_data_list.extend(batch.to_data_list())
        if max_batches is not None and (bidx + 1) >= max_batches:
            break
    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    from torch_geometric.data import Batch

    return {
        "frac_coords": frac_coords,
        "num_atoms": num_atoms,
        "atom_types": atom_types,
        "lattices": lattices,
        "lengths": lengths,
        "angles": angles,
        "input_data_batch": Batch.from_data_list(input_data_list),
    }


def structure_from_dict(d):
    try:
        if min(d["lengths"].tolist()) < 0:
            return None
        if np.isnan(d["lengths"]).any() or np.isnan(d["angles"]).any() or np.isnan(d["frac_coords"]).any():
            return None
        st = Structure(
            lattice=Lattice.from_parameters(*(d["lengths"].tolist() + d["angles"].tolist())),
            species=d["atom_types"],
            coords=d["frac_coords"],
            coords_are_cartesian=False,
        )
        if st.volume < 0.1:
            return None
        return st
    except Exception:
        return None


def eval_match(data, keys, multi_eval: bool):
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    if multi_eval:
        pred_lists = [
            get_crystals_list(
                data["frac_coords"][i],
                data["atom_types"][i],
                data["lengths"][i],
                data["angles"][i],
                data["num_atoms"][i],
            )
            for i in range(data["frac_coords"].shape[0])
        ]
        n = len(pred_lists[0])
        preds_per_gt = [
            [structure_from_dict(pred_lists[s][j]) for s in range(len(pred_lists))]
            for j in range(n)
        ]
    else:
        pred_list = get_crystals_list(
            data["frac_coords"][0],
            data["atom_types"][0],
            data["lengths"][0],
            data["angles"][0],
            data["num_atoms"][0],
        )
        n = len(pred_list)
        preds_per_gt = [[structure_from_dict(pred_list[j])] for j in range(n)]

    batch = data["input_data_batch"]
    gts = get_crystals_true_list(batch.frac_coords, batch.atom_types, batch.lattice_matrix, batch.num_atoms)
    gt_structs = [structure_from_dict(g) for g in gts]

    rows = []
    hits = 0
    for i, gt in enumerate(gt_structs):
        rid = keys[i] if i < len(keys) else str(i)
        best = None
        matched_k = None
        if gt is not None:
            for k, pred in enumerate(preds_per_gt[i], start=1):
                if pred is None:
                    continue
                try:
                    rms = matcher.get_rms_dist(pred, gt)
                except Exception:
                    rms = None
                if rms is not None:
                    rms0 = float(rms[0])
                    if best is None or rms0 < best:
                        best = rms0
                        matched_k = k
        hit = best is not None
        hits += int(hit)
        rows.append({"id": rid, "match": hit, "rms": best, "matched_k": matched_k})
    return hits, n, rows


def load_lmdb_keys(path: Path) -> list[str]:
    import lmdb

    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        keys = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k, _ in txn.cursor()]
    env.close()
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_evals", type=int, default=1)
    ap.add_argument("--label", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--onebatch", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    ckpts = list(model_path.glob("*.ckpt"))
    regular = [p for p in ckpts if "last_one" not in p.name]
    last = [p for p in ckpts if "last_one" in p.name]
    regular = sorted(regular, key=lambda x: int(x.name.split("step=")[1].split(".ckpt")[0]) if "step=" in x.name else 0)
    sorted_files = regular + last
    ckpt = sorted_files[args.label]
    print("ckpt", ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSPFlow.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.to(device)
    model.eval()
    print("keep_lattice", model.keep_lattice, "device", model.device)

    ds = CrystDataset_pxrd(path=args.test_data, gzip=True, conventional=False, xrd_augment=False)
    keys = [
        k.decode("utf-8") if isinstance(k, bytes) else str(k)
        for _, k in ds.cached_data.keys
    ]
    print("n_test", len(ds), "keys", len(keys))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    t0 = time.time()
    data = sample_loader(loader, model, args.num_evals, max_batches=1 if args.onebatch else None)
    elapsed = time.time() - t0
    print(f"sample time {elapsed:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"ckpt{args.label}_{args.num_evals}.pt"
    torch.save(data, pt_path)

    multi = args.num_evals > 1
    hits, n, rows = eval_match(data, keys, multi_eval=multi)
    summary = {
        "model_path": str(model_path),
        "ckpt": str(ckpt),
        "test_data": args.test_data,
        "num_evals": args.num_evals,
        "keep_lattice": bool(model.keep_lattice),
        "n": n,
        "hits": hits,
        "match_rate": hits / n if n else 0.0,
        "elapsed_s": elapsed,
        "onebatch": args.onebatch,
    }
    (out_dir / f"summary_n{args.num_evals}.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / f"per_sample_n{args.num_evals}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
