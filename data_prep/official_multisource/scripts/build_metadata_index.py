#!/usr/bin/env python
"""Build a metadata + dedup index over crystal-native JSONL splits (P2/P3).

For every record this emits one index row with the fields the training and
evaluation pipelines need for sampling weights, stratified evaluation and
length-grouped batching, plus a coarse near-duplicate group id so the corpus
can be de-duplicated without an O(N^2) StructureMatcher pass.

Columns
  id, source, split, n_sites, n_sites_bucket, space_group, crystal_system,
  n_elements, reduced_formula, vpa (volume/atom), a, b, c, alpha, beta, gamma,
  struct_hash, hull_tier, formation_energy, e_above_hull, tier_a,
  est_seq_len_v2, coarse_group, is_dup_keep

Dedup (P3): records sharing a coarse key land in one ``coarse_group``; one
representative per group is kept (``is_dup_keep=True``) using the source
priority MP > WBM > OQMD > Alexandria > AFLOW (ties broken by first
occurrence). Exact ``struct_hash`` equality is always a duplicate.

  --dedup-mode coarse (default): (reduced_formula, space_group, n_sites,
      round(vpa)). Fast but merges distinct polymorphs that share density.
  --dedup-mode tight: (reduced_formula, space_group, n_sites, sorted cell
      lengths on a --len-grid grid, sorted angles on a --angle-grid grid) --
      a cell-metric fingerprint that keeps polymorphs apart while still
      merging the same structure relaxed across different sources.

Outputs (to --out-dir)
  metadata_index.parquet (or .jsonl.gz if pyarrow is unavailable)
  metadata_index_summary.json
  dedup_keep_ids.txt / dedup_drop_ids.txt

Usage
  python build_metadata_index.py \
    --jsonl data_prep/outputs/crystal_native_realpxrd/train.jsonl \
    --jsonl data_prep/outputs/crystal_native_realpxrd/valid.jsonl \
    --jsonl data_prep/outputs/crystal_native_realpxrd/test.jsonl \
    --out-dir data_prep/outputs/metadata_index_realpxrd
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_PRIORITY = {"mp": 0, "wbm": 1, "oqmd": 2, "alexandria": 3, "aflow": 4}
DEFAULT_PRIORITY = 9


def reduced_formula_key(atom_types: list[str]) -> str:
    counts = Counter(atom_types)
    gcd = 0
    for n in counts.values():
        gcd = math.gcd(gcd, n)
    gcd = max(1, gcd)
    return "".join(f"{el}{counts[el] // gcd}" for el in sorted(counts))


def volume_per_atom(lattice_matrix: list[list[float]], n_sites: int) -> float:
    a, b, c = lattice_matrix
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return abs(det) / max(1, n_sites)


def lattice_metric(
    lattice_matrix: list[list[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Cell lengths (a, b, c) in Angstrom and angles (alpha, beta, gamma) in deg.

    Pure-python; works on numpy arrays or nested lists. Robust to degenerate
    rows (returns 0 for ill-defined angles).
    """
    a, b, c = lattice_matrix[0], lattice_matrix[1], lattice_matrix[2]

    def _norm(v) -> float:
        return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)

    def _dot(u, v) -> float:
        return float(u[0]) * float(v[0]) + float(u[1]) * float(v[1]) + float(u[2]) * float(v[2])

    la, lb, lc = _norm(a), _norm(b), _norm(c)

    def _angle(u, v, nu: float, nv: float) -> float:
        if nu <= 0.0 or nv <= 0.0:
            return 0.0
        cosv = max(-1.0, min(1.0, _dot(u, v) / (nu * nv)))
        return math.degrees(math.acos(cosv))

    alpha = _angle(b, c, lb, lc)
    beta = _angle(a, c, la, lc)
    gamma = _angle(a, b, la, lb)
    return (la, lb, lc), (alpha, beta, gamma)


def dedup_key(
    mode: str,
    fkey: str,
    sg: int,
    n_sites: int,
    vpa: float,
    lengths: tuple[float, float, float],
    angles: tuple[float, float, float],
    len_grid: float,
    angle_grid: float,
) -> tuple:
    """Coarse near-duplicate key.

    ``coarse``: (reduced_formula, space_group, n_sites, round(vpa)).
    ``tight``: (reduced_formula, space_group, n_sites, sorted-cell-length grid,
    sorted-angle grid) -- a cell-metric fingerprint that keeps distinct
    polymorphs apart while still merging the same structure across sources.
    """
    if mode == "tight":
        len_b = tuple(int(round(x / len_grid)) for x in sorted(lengths))
        ang_b = tuple(int(round(x / angle_grid)) for x in sorted(angles))
        return (fkey, sg, n_sites, len_b, ang_b)
    return (fkey, sg, n_sites, int(round(vpa)))


def crystal_system(sg: int) -> str:
    if sg <= 0:
        return "unknown"
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def n_sites_bucket(n: int) -> str:
    if n <= 8:
        return "<=8"
    if n <= 20:
        return "9-20"
    if n <= 40:
        return "21-40"
    if n <= 64:
        return "41-64"
    if n <= 128:
        return "65-128"
    return ">128"


def hull_tier(eah: float | None) -> str:
    if eah is None or not math.isfinite(eah):
        return "unknown"
    if eah <= 0.001:
        return "stable"
    if eah <= 0.1:
        return "lt0.1"
    return "gt0.1"


def compute_tier_a(source: str, eah, fe, n_sites: int) -> bool:
    """Tier-A clean-layer membership (per 总体实施方案 3.1).

    MP + WBM full; Alexandria/OQMD with e_above_hull <= 0.1; AFLOW with
    formation energy < 0; everything is dropped above 128 sites.
    """
    if n_sites > 128:
        return False
    s = source.lower()
    if s in ("mp", "wbm"):
        return True
    if s in ("alexandria", "oqmd"):
        return eah is not None and math.isfinite(float(eah)) and float(eah) <= 0.1
    if s == "aflow":
        return fe is not None and math.isfinite(float(fe)) and float(fe) < 0.0
    return False


def record_id(record: dict[str, Any]) -> str:
    rid = record.get("id")
    if rid:
        return str(rid)
    source = record.get("source")
    sid = record.get("source_id")
    if source and sid is not None:
        return f"{source}:{sid}"
    return str(record.get("structure_hash") or "")


def iter_records(path: Path, max_records: int | None):
    if path.suffix == ".lmdb":
        import sys as _sys  # noqa: PLC0415

        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.lmdb_utils import iter_lmdb  # noqa: PLC0415

        gen = iter_lmdb(str(path), compressed=True)
        try:
            for i, (_key, rec) in enumerate(gen):
                yield rec
                if max_records is not None and i + 1 >= max_records:
                    break
        finally:
            gen.close()
        return
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            if max_records is not None and i + 1 >= max_records:
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True, dest="jsonls")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-records", type=int, default=None,
                        help="cap records per file (debug)")
    parser.add_argument("--dedup-mode", choices=("coarse", "tight"), default="coarse",
                        help="coarse: formula+sg+n_sites+round(vpa); "
                             "tight: formula+sg+n_sites+cell-metric fingerprint")
    parser.add_argument("--len-grid", type=float, default=0.1,
                        help="tight mode: cell-length grid in Angstrom")
    parser.add_argument("--angle-grid", type=float, default=1.0,
                        help="tight mode: cell-angle grid in degrees")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols: dict[str, list] = {
        k: []
        for k in (
            "id", "source", "split", "n_sites", "n_sites_bucket", "space_group",
            "crystal_system", "n_elements", "reduced_formula", "vpa",
            "a", "b", "c", "alpha", "beta", "gamma",
            "struct_hash", "hull_tier", "formation_energy", "e_above_hull",
            "tier_a", "est_seq_len_v2", "coarse_group",
        )
    }

    # Coarse near-duplicate grouping + per-group best representative.
    group_of: dict[tuple, int] = {}
    hash_group: dict[str, int] = {}
    group_best: dict[int, tuple[int, int]] = {}  # gid -> (priority, row_index)
    next_gid = 0

    summary_counts: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_system: Counter[str] = Counter()
    by_bucket: Counter[str] = Counter()
    by_source: Counter[str] = Counter()

    t0 = time.time()
    row_index = 0
    for path_str in args.jsonls:
        path = Path(path_str)
        split = path.stem
        n_file = 0
        for rec in iter_records(path, args.max_records):
            raw_atoms = rec.get("atom_types")
            if raw_atoms is None:
                raw_atoms = rec.get("atom_type") or []
            atom_types = [str(a) for a in raw_atoms]
            n_sites = int(rec.get("n_sites") or len(atom_types))
            sg = int(rec.get("space_group") or 0)
            fkey = reduced_formula_key(atom_types)
            try:
                lat = rec["lattice_matrix"]
                vpa = float(volume_per_atom(lat, n_sites))
                lengths, angles = lattice_metric(lat)
            except Exception:
                vpa = 0.0
                lengths, angles = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            shash = str(rec.get("structure_hash") or "")
            source = str(rec.get("source") or "unknown")
            n_elem = len(set(atom_types))
            eah = rec.get("e_above_hull")
            fe = rec.get("formation_energy_per_atom")
            tier_a = compute_tier_a(source, eah, fe, n_sites)

            # group assignment: exact hash first, then coarse/tight key.
            gid = hash_group.get(shash) if shash else None
            if gid is None:
                ckey = dedup_key(
                    args.dedup_mode, fkey, sg, n_sites, vpa,
                    lengths, angles, args.len_grid, args.angle_grid,
                )
                gid = group_of.get(ckey)
                if gid is None:
                    gid = next_gid
                    next_gid += 1
                    group_of[ckey] = gid
                if shash:
                    hash_group[shash] = gid

            priority = SOURCE_PRIORITY.get(source.lower(), DEFAULT_PRIORITY)
            best = group_best.get(gid)
            if best is None or priority < best[0]:
                group_best[gid] = (priority, row_index)

            cols["id"].append(record_id(rec))
            cols["source"].append(source)
            cols["split"].append(split)
            cols["n_sites"].append(n_sites)
            cols["n_sites_bucket"].append(n_sites_bucket(n_sites))
            cols["space_group"].append(sg)
            cols["crystal_system"].append(crystal_system(sg))
            cols["n_elements"].append(n_elem)
            cols["reduced_formula"].append(fkey)
            cols["vpa"].append(round(vpa, 4))
            cols["a"].append(round(lengths[0], 4))
            cols["b"].append(round(lengths[1], 4))
            cols["c"].append(round(lengths[2], 4))
            cols["alpha"].append(round(angles[0], 2))
            cols["beta"].append(round(angles[1], 2))
            cols["gamma"].append(round(angles[2], 2))
            cols["struct_hash"].append(shash)
            cols["hull_tier"].append(hull_tier(float(eah) if eah is not None else None))
            cols["formation_energy"].append(round(float(fe), 4) if fe is not None else None)
            cols["e_above_hull"].append(round(float(eah), 4) if eah is not None else None)
            cols["tier_a"].append(tier_a)
            cols["est_seq_len_v2"].append(14 + 2 * n_elem + 4 * n_sites)
            cols["coarse_group"].append(gid)

            by_split[split] += 1
            by_system[crystal_system(sg)] += 1
            by_bucket[n_sites_bucket(n_sites)] += 1
            by_source[source] += 1
            row_index += 1
            n_file += 1
        print(f"[{time.time()-t0:.0f}s] {path.name}: {n_file} records "
              f"(total {row_index}, groups {next_gid})", flush=True)

    # is_dup_keep: True for the chosen representative of each coarse group.
    keep_rows = {row for _, row in group_best.values()}
    is_keep = [i in keep_rows for i in range(row_index)]
    cols["is_dup_keep"] = is_keep

    n_keep = sum(is_keep)
    n_drop = row_index - n_keep
    summary_counts["records"] = row_index
    summary_counts["coarse_groups"] = next_gid
    summary_counts["kept"] = n_keep
    summary_counts["dropped"] = n_drop

    # Tier-A clean layer: members, and members surviving dedup (the corpus that
    # Stage B actually trains on).
    tier_a_col = cols["tier_a"]
    src_col = cols["source"]
    tier_a_total = sum(1 for t in tier_a_col if t)
    tier_a_kept = sum(1 for i in range(row_index) if tier_a_col[i] and is_keep[i])
    tier_a_by_source: Counter[str] = Counter()
    tier_a_kept_by_source: Counter[str] = Counter()
    for i in range(row_index):
        if tier_a_col[i]:
            tier_a_by_source[src_col[i]] += 1
            if is_keep[i]:
                tier_a_kept_by_source[src_col[i]] += 1
    summary_counts["tier_a"] = tier_a_total
    summary_counts["tier_a_kept_after_dedup"] = tier_a_kept

    # Write index (parquet preferred, gzip-jsonl fallback).
    index_path, index_format = _write_index(out_dir, cols)

    keep_ids = [cols["id"][i] for i in range(row_index) if is_keep[i]]
    drop_ids = [cols["id"][i] for i in range(row_index) if not is_keep[i]]
    (out_dir / "dedup_keep_ids.txt").write_text("\n".join(keep_ids) + "\n", encoding="utf-8")
    (out_dir / "dedup_drop_ids.txt").write_text("\n".join(drop_ids) + "\n", encoding="utf-8")

    summary = {
        "inputs": [str(p) for p in args.jsonls],
        "index_path": str(index_path),
        "index_format": index_format,
        "dedup_mode": args.dedup_mode,
        "len_grid_ang": args.len_grid if args.dedup_mode == "tight" else None,
        "angle_grid_deg": args.angle_grid if args.dedup_mode == "tight" else None,
        "counts": dict(summary_counts),
        "dup_rate": n_drop / max(1, row_index),
        "by_split": dict(by_split),
        "by_source": dict(by_source),
        "by_crystal_system": dict(by_system),
        "by_n_sites_bucket": dict(by_bucket),
        "tier_a_by_source": dict(tier_a_by_source),
        "tier_a_kept_by_source": dict(tier_a_kept_by_source),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "metadata_index_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _write_index(out_dir: Path, cols: dict[str, list]) -> tuple[Path, str]:
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        table = pa.table(cols)
        path = out_dir / "metadata_index.parquet"
        pq.write_table(table, path, compression="zstd")
        return path, "parquet"
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"pyarrow unavailable ({exc}); writing gzip jsonl", flush=True)
        path = out_dir / "metadata_index.jsonl.gz"
        keys = list(cols.keys())
        n = len(cols["id"])
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for i in range(n):
                handle.write(json.dumps({k: cols[k][i] for k in keys}, ensure_ascii=False) + "\n")
        return path, "jsonl.gz"


if __name__ == "__main__":
    raise SystemExit(main())
