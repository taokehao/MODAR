#!/usr/bin/env python
"""Statistical analysis for multimodal ladder evaluation details.

Outputs:
  - Top-1 rate with bootstrap confidence intervals.
  - Paired Top-1 differences with bootstrap confidence intervals.
  - Top-1 decomposition by metadata groups (source / crystal_system), focused on
    neutron and XPS gains.

The script reads evaluation detail JSONL files produced by
``evaluate_generation.py``. It only extracts ``id`` and reranked Top-1
``structure_match`` from each row, so the downstream stats are paired by the
same test records across condition rungs.
"""

from __future__ import annotations

import argparse
import glob as globlib
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RUNGS = [
    "pxrd",
    "pxrd_element",
    "pxrd_xrf",
    "pxrd_neutron",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
]

FOCUS_PAIRS = [
    ("neutron_vs_pxrd", "pxrd_neutron", "pxrd"),
    ("xrf_vs_pxrd", "pxrd_xrf", "pxrd"),
    ("neutron_given_xrf", "pxrd_neutron_xrf", "pxrd_xrf"),
    ("xps_given_neutron_xrf", "pxrd_neutron_xrf_xps", "pxrd_neutron_xrf"),
    ("multi_vs_formula", "pxrd_neutron_xrf_xps", "pxrd_formula"),
    ("neutron_xrf_vs_formula", "pxrd_neutron_xrf", "pxrd_formula"),
]


def detail_paths(eval_dir: Path, split: str, cond: str, lat: int) -> list[str]:
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr1_start*_n*_s*.jsonl"
    paths = [p for p in globlib.glob(str(eval_dir / pattern)) if "_summary" not in p]
    paths.sort(key=lambda p: int(re.search(r"start(\d+)", p).group(1)) if re.search(r"start(\d+)", p) else 0)
    return paths


def row_top1(row: dict[str, Any]) -> bool:
    candidates = row.get("candidates") or []
    order = row.get("rerank_order") or list(range(len(candidates)))
    if not order:
        return False
    idx = order[0]
    if idx >= len(candidates):
        return False
    return bool(candidates[idx].get("structure_match"))


def load_hits(eval_dir: Path, split: str, rungs: list[str]) -> dict[str, dict[str, dict[str, bool]]]:
    """Return lattice -> rung -> id -> top1_hit."""
    out: dict[str, dict[str, dict[str, bool]]] = {"lattice_free": {}, "known_lattice": {}}
    for lat, lname in ((0, "lattice_free"), (1, "known_lattice")):
        for cond in rungs:
            hits: dict[str, bool] = {}
            for path in detail_paths(eval_dir, split, cond, lat):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if "target_error" in row:
                            continue
                        rid = str(row.get("id"))
                        if rid and rid != "None" and rid not in hits:
                            hits[rid] = row_top1(row)
            out[lname][cond] = hits
    return out


def collect_ids(hits: dict[str, dict[str, dict[str, bool]]]) -> set[str]:
    ids: set[str] = set()
    for by_rung in hits.values():
        for rows in by_rung.values():
            ids.update(rows)
    return ids


def load_metadata(index_path: Path, keep_ids: set[str]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    opener = gzip.open if index_path.suffix == ".gz" else open
    with opener(index_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rid = str(row.get("id"))
            if rid in keep_ids:
                meta[rid] = {
                    "source": str(row.get("source") or rid.split(":", 1)[0] or "unknown"),
                    "crystal_system": str(row.get("crystal_system") or "unknown"),
                }
                if len(meta) >= len(keep_ids):
                    break
    # Fallback source from id prefix if any id is absent from metadata.
    for rid in keep_ids - set(meta):
        meta[rid] = {"source": rid.split(":", 1)[0] or "unknown", "crystal_system": "unknown"}
    return meta


def bootstrap_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if n_boot <= 0 or len(x) == 1:
        return mean, mean, mean
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return mean, float(lo), float(hi)


def rate_ci(rows: dict[str, bool], ids: list[str], n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    x = np.asarray([rows[i] for i in ids if i in rows], dtype=float)
    mean, lo, hi = bootstrap_mean(x, n_boot, rng)
    return {"n": int(len(x)), "rate": mean, "lo": lo, "hi": hi}


def paired_diff_ci(
    a: dict[str, bool],
    b: dict[str, bool],
    ids: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    common = [i for i in ids if i in a and i in b]
    x = np.asarray([float(a[i]) - float(b[i]) for i in common], dtype=float)
    mean, lo, hi = bootstrap_mean(x, n_boot, rng)
    return {"n": int(len(x)), "diff": mean, "lo": lo, "hi": hi}


def fmt_pct(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{100 * x:.2f}"


def make_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Tier-A 10k Ladder Statistical Analysis")
    lines.append("")
    lines.append("All rates are reranked Top-1 (%). Bootstrap intervals are 95% CIs.")
    lines.append("")
    for lname in ("lattice_free", "known_lattice"):
        lines.append(f"## {lname}: Top-1 CI")
        lines.append("")
        lines.append("| rung | N | T1 | 95% CI |")
        lines.append("|---|---:|---:|---:|")
        for rung, row in results["rates"][lname].items():
            lines.append(
                f"| {rung} | {row['n']} | {fmt_pct(row['rate'])} | "
                f"[{fmt_pct(row['lo'])}, {fmt_pct(row['hi'])}] |"
            )
        lines.append("")
        lines.append(f"## {lname}: Paired Diffs")
        lines.append("")
        lines.append("| comparison | N | ΔT1 | 95% CI |")
        lines.append("|---|---:|---:|---:|")
        for label, row in results["paired_diffs"][lname].items():
            lines.append(
                f"| {label} | {row['n']} | {fmt_pct(row['diff'])} | "
                f"[{fmt_pct(row['lo'])}, {fmt_pct(row['hi'])}] |"
            )
        lines.append("")
    for group in ("crystal_system", "source"):
        lines.append(f"## Group Decomposition: {group}")
        lines.append("")
        lines.append(
            "| lattice | group | N | pxrd | +neutron | +xrf | +neutron+xrf | "
            "+xps | oracle | Δneutron | Δneutron|xrf | Δxps | Δmulti-oracle |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in results["groups"][group]:
            lines.append(
                f"| {row['lattice']} | {row['group']} | {row['n']} | "
                f"{fmt_pct(row['pxrd'])} | {fmt_pct(row['pxrd_neutron'])} | "
                f"{fmt_pct(row['pxrd_xrf'])} | {fmt_pct(row['pxrd_neutron_xrf'])} | "
                f"{fmt_pct(row['pxrd_neutron_xrf_xps'])} | {fmt_pct(row['pxrd_formula'])} | "
                f"{fmt_pct(row['d_neutron'])} | {fmt_pct(row['d_neutron_xrf'])} | "
                f"{fmt_pct(row['d_xps'])} | {fmt_pct(row['d_multi_oracle'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--metadata-index", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rungs", default=",".join(DEFAULT_RUNGS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-group-n", type=int, default=50)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    rungs = [x.strip() for x in args.rungs.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    hits = load_hits(eval_dir, args.split, rungs)
    all_ids = collect_ids(hits)
    meta = load_metadata(Path(args.metadata_index), all_ids)

    results: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "split": args.split,
        "rungs": rungs,
        "n_boot": args.n_boot,
        "rates": {"lattice_free": {}, "known_lattice": {}},
        "paired_diffs": {"lattice_free": {}, "known_lattice": {}},
        "groups": {"crystal_system": [], "source": []},
    }

    for lname in ("lattice_free", "known_lattice"):
        ids = sorted(set.intersection(*(set(hits[lname][r]) for r in rungs)))
        for rung in rungs:
            results["rates"][lname][rung] = rate_ci(hits[lname][rung], ids, args.n_boot, rng)
        for label, a, b in FOCUS_PAIRS:
            if a in hits[lname] and b in hits[lname]:
                results["paired_diffs"][lname][label] = paired_diff_ci(
                    hits[lname][a], hits[lname][b], ids, args.n_boot, rng
                )

    group_names: dict[str, set[str]] = {"crystal_system": set(), "source": set()}
    for rid in all_ids:
        for g in group_names:
            group_names[g].add(meta[rid].get(g, "unknown"))

    needed = ["pxrd", "pxrd_neutron", "pxrd_xrf", "pxrd_neutron_xrf", "pxrd_neutron_xrf_xps", "pxrd_formula"]
    for group in ("crystal_system", "source"):
        for lname in ("lattice_free", "known_lattice"):
            common_ids = sorted(set.intersection(*(set(hits[lname][r]) for r in needed if r in hits[lname])))
            by_group: dict[str, list[str]] = defaultdict(list)
            for rid in common_ids:
                by_group[meta[rid].get(group, "unknown")].append(rid)
            for gname, gids in sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                if len(gids) < args.min_group_n:
                    continue
                vals = {r: float(np.mean([hits[lname][r][rid] for rid in gids])) for r in needed}
                row = {
                    "lattice": lname,
                    "group": gname,
                    "n": len(gids),
                    **vals,
                    "d_neutron": vals["pxrd_neutron"] - vals["pxrd"],
                    "d_neutron_xrf": vals["pxrd_neutron_xrf"] - vals["pxrd_xrf"],
                    "d_xps": vals["pxrd_neutron_xrf_xps"] - vals["pxrd_neutron_xrf"],
                    "d_multi_oracle": vals["pxrd_neutron_xrf_xps"] - vals["pxrd_formula"],
                }
                results["groups"][group].append(row)

    (out_dir / "ladder_stats.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = make_markdown(results)
    (out_dir / "ladder_stats.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nwrote {out_dir / 'ladder_stats.json'}")
    print(f"wrote {out_dir / 'ladder_stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
