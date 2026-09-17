#!/usr/bin/env python
"""Train <-> eval near-duplicate (leakage) audit for crystal-native JSONL splits.

Two-stage protocol (per 总体实施方案 P1):
  1. coarse key  (reduced_formula, space_group, n_sites) + volume/atom within
     ±2 % relative tolerance;
  2. StructureMatcher confirmation on the coarse candidates.

Exact ``structure_hash`` equality short-circuits stage 2.

Outputs (to --out-dir):
  leakage_<eval-name>_details.jsonl   per contaminated eval record: matched train ids
  leakage_<eval-name>_summary.json    rates + parameters
  contaminated_train_ids_<eval-name>.txt   train ids to drop before SFT

Usage:
  python leakage_audit.py \
    --train data_prep/outputs/crystal_native_mp_wbm/train.jsonl \
    --eval  data_prep/outputs/crystal_native_mp_wbm/valid.jsonl \
    --eval  data_prep/outputs/crystal_native_mp_wbm/test.jsonl \
    --out-dir data_prep/outputs/leakage_audit_mp_wbm
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

VPA_REL_TOL = 0.02
MATCHER_KWARGS = {"ltol": 0.2, "stol": 0.3, "angle_tol": 5.0}


def reduced_formula_key(atom_types: list[str]) -> str:
    counts = Counter(atom_types)
    gcd = 0
    for n in counts.values():
        gcd = math.gcd(gcd, n)
    gcd = max(1, gcd)
    return "|".join(f"{el}:{counts[el] // gcd}" for el in sorted(counts))


def volume_per_atom(lattice_matrix: list[list[float]], n_sites: int) -> float:
    a, b, c = lattice_matrix
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return abs(det) / max(1, n_sites)


def coarse_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        reduced_formula_key([str(a) for a in record["atom_types"]]),
        int(record.get("space_group") or 0),
        int(record.get("n_sites") or len(record["atom_types"])),
    )


def index_train(path: Path) -> dict[tuple, list[tuple[int, str, float, str]]]:
    """Map coarse key -> list of (byte_offset, id, vpa, structure_hash)."""
    index: dict[tuple, list[tuple[int, str, float, str]]] = {}
    with path.open("rb") as handle:
        offset = handle.tell()
        for raw in iter(handle.readline, b""):
            line = raw.strip()
            if line:
                rec = json.loads(line)
                key = coarse_key(rec)
                vpa = volume_per_atom(rec["lattice_matrix"], key[2])
                index.setdefault(key, []).append(
                    (offset, str(rec.get("id")), vpa, str(rec.get("structure_hash") or ""))
                )
            offset = handle.tell()
    return index


def read_records_at(path: Path, offsets: list[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for offset in sorted(set(offsets)):
            handle.seek(offset)
            out[offset] = json.loads(handle.readline())
    return out


def _structure(record: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice(record["lattice_matrix"]),
        [str(a) for a in record["atom_types"]],
        record["frac_coords"],
    )


def _confirm_pairs(args: tuple[dict[str, Any], list[dict[str, Any]]]) -> list[str]:
    """Worker: return train ids whose structures match the eval record."""
    eval_rec, candidates = args
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(**MATCHER_KWARGS)
    confirmed: list[str] = []
    try:
        eval_struct = _structure(eval_rec)
    except Exception:
        return confirmed
    for cand in candidates:
        try:
            if matcher.fit(eval_struct, _structure(cand)):
                confirmed.append(str(cand.get("id")))
        except Exception:
            continue
    return confirmed


def audit_one(
    train_path: Path,
    train_index: dict[tuple, list[tuple[int, str, float, str]]],
    eval_path: Path,
    out_dir: Path,
    workers: int,
) -> dict[str, Any]:
    eval_name = eval_path.stem
    t0 = time.time()

    eval_records: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                eval_records.append(json.loads(line))

    # Stage 1: coarse candidates.
    needed_offsets: list[int] = []
    coarse_hits: list[tuple[int, list[tuple[int, str, float, str]], list[str]]] = []
    for i, rec in enumerate(eval_records):
        key = coarse_key(rec)
        bucket = train_index.get(key)
        if not bucket:
            continue
        vpa = volume_per_atom(rec["lattice_matrix"], key[2])
        eval_hash = str(rec.get("structure_hash") or "")
        cands = [c for c in bucket if abs(c[2] - vpa) <= VPA_REL_TOL * max(vpa, 1e-9)]
        if not cands:
            continue
        exact = [c[1] for c in cands if eval_hash and c[3] == eval_hash]
        to_confirm = [c for c in cands if not (eval_hash and c[3] == eval_hash)]
        coarse_hits.append((i, to_confirm, exact))
        needed_offsets.extend(c[0] for c in to_confirm)

    train_records = read_records_at(train_path, needed_offsets) if needed_offsets else {}

    # Stage 2: StructureMatcher confirmation.
    jobs = []
    job_meta = []
    for i, to_confirm, exact in coarse_hits:
        cands = [train_records[c[0]] for c in to_confirm if c[0] in train_records]
        if cands:
            jobs.append((eval_records[i], cands))
            job_meta.append(i)

    confirmed_by_record: dict[int, list[str]] = {}
    if jobs:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, ids in zip(job_meta, pool.map(_confirm_pairs, jobs, chunksize=8)):
                    if ids:
                        confirmed_by_record[i] = ids
        else:
            for i, job in zip(job_meta, jobs):
                ids = _confirm_pairs(job)
                if ids:
                    confirmed_by_record[i] = ids

    # Merge exact-hash hits.
    details: list[dict[str, Any]] = []
    contaminated_train_ids: set[str] = set()
    n_coarse = 0
    n_leaked = 0
    coarse_index = {i: (to_confirm, exact) for i, to_confirm, exact in coarse_hits}
    for i, rec in enumerate(eval_records):
        hit = coarse_index.get(i)
        if hit is None:
            continue
        n_coarse += 1
        to_confirm, exact = hit
        matched = list(exact) + confirmed_by_record.get(i, [])
        if not matched:
            continue
        n_leaked += 1
        contaminated_train_ids.update(matched)
        details.append(
            {
                "eval_id": rec.get("id"),
                "eval_source": rec.get("source"),
                "n_sites": rec.get("n_sites"),
                "exact_hash_matches": exact,
                "matcher_confirmed": confirmed_by_record.get(i, []),
                "n_coarse_candidates": len(to_confirm) + len(exact),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / f"leakage_{eval_name}_details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ids_path = out_dir / f"contaminated_train_ids_{eval_name}.txt"
    ids_path.write_text("\n".join(sorted(contaminated_train_ids)) + "\n", encoding="utf-8")

    summary = {
        "train": str(train_path),
        "eval": str(eval_path),
        "n_eval_records": len(eval_records),
        "n_coarse_hit_records": n_coarse,
        "n_leaked_records": n_leaked,
        "leak_rate": n_leaked / max(1, len(eval_records)),
        "n_contaminated_train_ids": len(contaminated_train_ids),
        "vpa_rel_tol": VPA_REL_TOL,
        "matcher": MATCHER_KWARGS,
        "elapsed_sec": round(time.time() - t0, 1),
        "details_path": str(details_path),
        "train_ids_path": str(ids_path),
    }
    (out_dir / f"leakage_{eval_name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", action="append", required=True, dest="evals")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train_path = Path(args.train)
    print(f"indexing train: {train_path}", flush=True)
    t0 = time.time()
    train_index = index_train(train_path)
    print(
        f"train index built: {sum(len(v) for v in train_index.values())} records, "
        f"{len(train_index)} coarse keys, {time.time() - t0:.0f}s",
        flush=True,
    )

    for eval_path in args.evals:
        summary = audit_one(
            train_path, train_index, Path(eval_path), Path(args.out_dir), args.workers
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
