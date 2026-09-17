#!/usr/bin/env python
"""Offline Rietveld-lite refine + rerank ablation on frozen noisy pools.

Compares deploy match (σ=0.40 ±0.20°) with/without lattice refine on top-M
candidates. Rebuilds noisy reference peaks with the same seed as eval.

Usage
  python crystal_language/refine_rerank_ablation.py \\
    --eval-dir crystal_language/eval_results/ladder_tierA_s2_noisy_exp_500 \\
    --data-dir data_prep/outputs/crystal_language_unified_tierA_v2 \\
    --conditions pxrd_neutron_xrf --lattices 0 \\
    --refine-top-m 20 --workers 16 \\
    --out crystal_language/eval_results/refine_rerank_ablation_base.json
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [str(PACKAGE_ROOT)] + [
    p for p in sys.path if Path(p).resolve() != SCRIPT_DIR
]

from crystal_language.evaluate_generation import peaks_from_dense_spectrum  # noqa: E402
from crystal_language.generate import build_prompt_v2  # noqa: E402
from crystal_language.noise import rasterize_noisy_spectrum  # noqa: E402
from crystal_language.refine import RefineConfig, refine_candidate, score_without_refine  # noqa: E402
from crystal_language.rerank import structure_dedup_key  # noqa: E402

TOP_KS = [1, 5, 10, 20, 50]


def topk_any(flags: list[bool], k: int) -> bool:
    return any(flags[:k])


def load_pool(eval_dir: Path, split: str, cond: str, lat: int) -> list[dict]:
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr1_start*_n*_s*.jsonl"
    # bigcell uses different naming
    paths = [p for p in globlib.glob(str(eval_dir / pattern)) if "_summary" not in p]
    if not paths:
        pattern2 = f"{split}_*_lat{lat}_*_rr1_*.jsonl"
        paths = [p for p in globlib.glob(str(eval_dir / pattern2)) if "_summary" not in p]
        # filter condition in name if present
        paths = [p for p in paths if cond in Path(p).name]
    paths.sort(
        key=lambda p: int(m.group(1))
        if (m := re.search(r"start(\d+)", p))
        else 0
    )
    seen: set[Any] = set()
    records: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "target_error" in row:
                    continue
                idx = row.get("index")
                if idx is not None:
                    if idx in seen:
                        continue
                    seen.add(idx)
                records.append(
                    {
                        "index": int(idx) if idx is not None else len(records),
                        "id": row.get("id"),
                        "candidates": row.get("candidates") or [],
                    }
                )
    return records


def load_source_records(data_dir: Path, split: str, indices: set[int]) -> dict[int, dict]:
    path = data_dir / f"{split}.jsonl"
    want = set(indices)
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i not in want:
                continue
            rec = json.loads(line)
            rec["_source_index"] = i
            out[i] = rec
            if len(out) >= len(want):
                break
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"missing {len(missing)} source indices, e.g. {sorted(missing)[:5]}")
    return out


def rebuild_noisy_refs(
    record: dict,
    condition_set: str,
    use_lattice: bool,
    noise_seed: int,
    spectra_slots: int = 16,
) -> dict[str, Any]:
    import torch

    rng = random.Random(noise_seed + int(record["_source_index"]))
    _, used = build_prompt_v2(
        record,
        condition_set,
        use_lattice,
        spectra_slots,
        rng=rng,
        noise_profile="experimental",
    )
    spec = record.get("spectra") or {}
    peaks: dict[str, list[tuple[float, float]] | None] = {"pxrd": None, "neutron": None}
    for name in ("pxrd", "neutron"):
        if not used.get(name):
            continue
        vec = rasterize_noisy_spectrum(
            spec.get(f"{name}_x") or [],
            spec.get(f"{name}_y") or [],
            name=name,
            rng=rng,
            profile="experimental",
        )
        peaks[name] = peaks_from_dense_spectrum(torch.as_tensor(vec.detach().cpu()))
    return {"peaks": peaks, "used": used}


def _order_flags(
    candidates: list[dict],
    scores: dict[Any, float],
) -> list[bool]:
    compact = []
    for c in candidates:
        struct = c.get("structure")
        if struct is None:
            compact.append({"scorable": False, "sm": bool(c.get("structure_match"))})
            continue
        compact.append(
            {
                "scorable": True,
                "sm": bool(c.get("structure_match")),
                "key": structure_dedup_key(struct),
            }
        )
    groups: dict[Any, list[int]] = {}
    for i, c in enumerate(compact):
        if c["scorable"]:
            groups.setdefault(c["key"], []).append(i)
    reps = [members[0] for members in groups.values()]
    uniq = sorted(reps, key=lambda i: (-(scores.get(compact[i]["key"], float("-inf"))), i))
    dups = sorted(
        [m for members in groups.values() for m in members[1:]],
        key=lambda i: (-(scores.get(compact[i]["key"], float("-inf"))), i),
    )
    invalid = [i for i, c in enumerate(compact) if not c["scorable"]]
    order = uniq + dups + invalid
    return [compact[i]["sm"] for i in order]


def _worker_score_job(job: dict) -> dict:
    """Process one unique structure: unreined + optional refine."""
    cfg = RefineConfig(**job["cfg"])
    payload = job["payload"]
    ref_pxrd = job["ref_pxrd"]
    ref_nd = job["ref_nd"]
    base = score_without_refine(payload, ref_pxrd, ref_nd, cfg=cfg)
    out = {
        "key": job["key"],
        "base_score": base.score,
        "base_pxrd": base.pxrd_cos,
        "refine_score": base.score,
        "refine_pxrd": base.pxrd_cos,
        "scale": 1.0,
    }
    if job.get("do_refine"):
        refined = refine_candidate(payload, ref_pxrd, ref_nd, cfg=cfg)
        out["refine_score"] = refined.score
        out["refine_pxrd"] = refined.pxrd_cos
        out["scale"] = refined.scale
    return out


def eval_condition(
    pool: list[dict],
    source: dict[int, dict],
    *,
    condition: str,
    use_lattice: bool,
    noise_seed: int,
    cfg: RefineConfig,
    refine_top_m: int,
    workers: int,
) -> dict:
    base_flags: list[list[bool]] = []
    refine_flags: list[list[bool]] = []
    n_refined = 0
    n_scale_moved = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rec_i, rec in enumerate(pool):
            src = source[rec["index"]]
            refs = rebuild_noisy_refs(src, condition, use_lattice, noise_seed)
            ref_pxrd = refs["peaks"]["pxrd"]
            ref_nd = refs["peaks"]["neutron"]

            groups: dict[Any, dict] = {}
            for c in rec["candidates"]:
                struct = c.get("structure")
                if struct is None:
                    continue
                key = structure_dedup_key(struct)
                if key not in groups:
                    groups[key] = struct

            # Pass 1: unreined scores for all unique structures.
            pre_jobs = [
                {
                    "key": key,
                    "payload": payload,
                    "ref_pxrd": ref_pxrd,
                    "ref_nd": ref_nd,
                    "cfg": cfg.__dict__,
                    "do_refine": False,
                }
                for key, payload in groups.items()
            ]
            pre = list(ex.map(_worker_score_job, pre_jobs, chunksize=8))
            base_scores = {r["key"]: r["base_score"] for r in pre}
            ranked = sorted(base_scores.keys(), key=lambda k: (-base_scores[k], str(k)))
            refine_set = set(ranked[:refine_top_m])

            # Pass 2: refine only top-M.
            refine_jobs = [
                {
                    "key": key,
                    "payload": groups[key],
                    "ref_pxrd": ref_pxrd,
                    "ref_nd": ref_nd,
                    "cfg": cfg.__dict__,
                    "do_refine": True,
                }
                for key in refine_set
            ]
            refined = list(ex.map(_worker_score_job, refine_jobs, chunksize=2)) if refine_jobs else []
            refine_scores = dict(base_scores)
            for r in refined:
                refine_scores[r["key"]] = r["refine_score"]
                n_refined += 1
                if abs(r.get("scale", 1.0) - 1.0) > 1e-6:
                    n_scale_moved += 1

            base_flags.append(_order_flags(rec["candidates"], base_scores))
            refine_flags.append(_order_flags(rec["candidates"], refine_scores))

            if (rec_i + 1) % 25 == 0 or rec_i == 0:
                b1 = 100 * sum(topk_any(f, 1) for f in base_flags) / len(base_flags)
                r1 = 100 * sum(topk_any(f, 1) for f in refine_flags) / len(refine_flags)
                print(
                    f"  [{condition} lat={int(use_lattice)}] {rec_i+1}/{len(pool)}  "
                    f"baseT1={b1:.1f} refineT1={r1:.1f}",
                    flush=True,
                )

    def metrics(all_flags: list[list[bool]]) -> dict:
        n = max(1, len(all_flags))
        return {
            f"T{k}": round(100.0 * sum(int(topk_any(f, k)) for f in all_flags) / n, 2)
            for k in TOP_KS
        } | {"N": len(all_flags)}

    return {
        "condition": condition,
        "use_lattice": use_lattice,
        "baseline_deploy": metrics(base_flags),
        "refine_top_m": metrics(refine_flags),
        "delta_T1": round(
            metrics(refine_flags)["T1"] - metrics(base_flags)["T1"], 2
        ),
        "n_refined_structs": n_refined,
        "n_scale_moved": n_scale_moved,
        "cfg": cfg.__dict__,
        "refine_top_m_param": refine_top_m,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", default="pxrd_neutron_xrf")
    ap.add_argument("--lattices", default="0")
    ap.add_argument("--noise-seed", type=int, default=20260706)
    ap.add_argument("--refine-top-m", type=int, default=20)
    ap.add_argument("--scale-max", type=float, default=0.03)
    ap.add_argument("--n-scales", type=int, default=7)
    ap.add_argument("--axis-max", type=float, default=0.0)
    ap.add_argument("--n-axis", type=int, default=5)
    ap.add_argument("--match-sigma", type=float, default=0.25,
                    help="raster σ used for scoring/refine (sharper than deploy 0.40 helps aniso)")
    ap.add_argument("--shift-max", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = RefineConfig(
        scale_max=args.scale_max,
        n_scales=args.n_scales,
        axis_max=args.axis_max,
        n_axis=args.n_axis,
        match_sigma=args.match_sigma,
        shift_max=args.shift_max,
    )
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    lattices = [int(x) for x in args.lattices.split(",") if x.strip() != ""]
    eval_dir = Path(args.eval_dir)
    data_dir = Path(args.data_dir)

    results = []
    for lat in lattices:
        for cond in conditions:
            print(f"=== refine ablation: {cond} lat={lat} ===", flush=True)
            pool = load_pool(eval_dir, args.split, cond, lat)
            if args.max_records is not None:
                pool = pool[: args.max_records]
            source = load_source_records(data_dir, args.split, {r["index"] for r in pool})
            row = eval_condition(
                pool,
                source,
                condition=cond,
                use_lattice=bool(lat),
                noise_seed=args.noise_seed,
                cfg=cfg,
                refine_top_m=args.refine_top_m,
                workers=args.workers,
            )
            results.append(row)
            print(
                f"  DONE T1 {row['baseline_deploy']['T1']} -> {row['refine_top_m']['T1']} "
                f"(Δ={row['delta_T1']:+.2f})",
                flush=True,
            )

    out = {
        "results": results,
        "mean_delta_T1": round(sum(r["delta_T1"] for r in results) / max(1, len(results)), 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
