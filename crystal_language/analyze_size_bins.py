#!/usr/bin/env python
"""Aggregate ladder evaluation details by target atom-count bins."""

from __future__ import annotations

import argparse
import glob as globlib
import json
import re
from pathlib import Path
from typing import Any


TOP_KS = (1, 5, 10, 20, 50)
DEFAULT_RUNGS = (
    "pxrd",
    "pxrd_neutron",
    "pxrd_xrf",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
)
RUNG_LABELS = {
    "pxrd": "PXRD only",
    "pxrd_neutron": "PXRD + neutron",
    "pxrd_xrf": "PXRD + XRF",
    "pxrd_neutron_xrf": "PXRD + neutron + XRF",
    "pxrd_neutron_xrf_xps": "PXRD + neutron + XRF + XPS-valence",
    "pxrd_formula": "PXRD + exact formula (oracle)",
}


def topk_any(candidates: list[dict[str, Any]], key: str, k: int) -> bool:
    return any(bool(c.get(key)) for c in candidates[:k])


def start_key(path: str) -> int:
    m = re.search(r"_start(\d+)_", Path(path).name)
    return int(m.group(1)) if m else 0


def empty_bin(label: str) -> dict[str, Any]:
    return {
        "bin": label,
        "n_records": 0,
        "comp_hits": 0,
        "topk_hits": {str(k): 0 for k in TOP_KS},
    }


def finalize_bin(row: dict[str, Any]) -> dict[str, Any]:
    n = int(row["n_records"])
    return {
        "bin": row["bin"],
        "n_records": n,
        "comp_rate": row["comp_hits"] / max(1, n),
        "topk_rerank": {
            str(k): row["topk_hits"][str(k)] / max(1, n)
            for k in TOP_KS
        },
    }


def aggregate_config(eval_dir: Path, split: str, cond: str, lat: int, threshold: int) -> dict[str, Any]:
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr*_start*_n*_s*.jsonl"
    paths = [
        p for p in globlib.glob(str(eval_dir / pattern))
        if "_summary" not in p
    ]
    paths.sort(key=start_key)

    bins = {
        f"<={threshold}": empty_bin(f"<={threshold}"),
        f">{threshold}": empty_bin(f">{threshold}"),
    }
    seen: set[int] = set()
    skipped_missing_size = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = row.get("index")
                if idx is not None:
                    if int(idx) in seen:
                        continue
                    seen.add(int(idx))
                n_sites = row.get("target_n_sites")
                if n_sites is None:
                    skipped_missing_size += 1
                    continue
                bin_key = f"<={threshold}" if int(n_sites) <= threshold else f">{threshold}"
                bucket = bins[bin_key]
                cands = row.get("candidates") or []
                order = row.get("rerank_order")
                reranked = [cands[i] for i in order if i < len(cands)] if order else cands

                bucket["n_records"] += 1
                bucket["comp_hits"] += int(topk_any(cands, "composition_match", max(TOP_KS)))
                for k in TOP_KS:
                    bucket["topk_hits"][str(k)] += int(topk_any(reranked, "structure_match", k))

    return {
        "condition_set": cond,
        "use_lattice": lat,
        "shards": len(paths),
        "skipped_missing_size": skipped_missing_size,
        "bins": {k: finalize_bin(v) for k, v in bins.items()},
    }


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def write_markdown(result: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    threshold = result["threshold"]
    lines.append(f"# Ladder Size Bins (threshold={threshold} atoms)")
    lines.append("")
    lines.append("Metric: rerank Top-k (%), grouped by `target_n_sites`.")
    lines.append("")
    for lat_name in ("lattice_free", "known_lattice"):
        lines.append(f"## {lat_name}")
        lines.append("")
        lines.append("| bin | rung | N | comp | T1 | T5 | T10 | T20 | T50 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for bin_key in (f"<={threshold}", f">{threshold}"):
            for cond in result["rungs"]:
                row = result["table"][lat_name][cond]["bins"][bin_key]
                rr = row["topk_rerank"]
                lines.append(
                    f"| {bin_key} | {RUNG_LABELS.get(cond, cond)} | {row['n_records']} | "
                    f"{pct(row['comp_rate'])} | {pct(rr['1'])} | {pct(rr['5'])} | "
                    f"{pct(rr['10'])} | {pct(rr['20'])} | {pct(rr['50'])} |"
                )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--rungs", default=",".join(DEFAULT_RUNGS))
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rungs = [x.strip() for x in args.rungs.split(",") if x.strip()]
    table: dict[str, dict[str, Any]] = {"lattice_free": {}, "known_lattice": {}}
    for lat, lat_name in ((0, "lattice_free"), (1, "known_lattice")):
        for cond in rungs:
            table[lat_name][cond] = aggregate_config(eval_dir, args.split, cond, lat, args.threshold)

    result = {
        "eval_dir": str(eval_dir),
        "split": args.split,
        "threshold": args.threshold,
        "rungs": rungs,
        "table": table,
    }
    out_json = Path(args.out_json) if args.out_json else eval_dir / f"size_bins_{args.threshold}.json"
    out_md = Path(args.out_md) if args.out_md else eval_dir / f"size_bins_{args.threshold}.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, out_md)
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
