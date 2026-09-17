#!/usr/bin/env python3
"""Route SI eval rows by n_sites: mix model for n<13, 13–20 specialist otherwise.

Writes a merged eval jsonl (and optional energy jsonl) so analyze_rps_step1.py
can score the routed pool without pooling extra temperatures or N>100.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_eval(eval_dir: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    paths = sorted(eval_dir.glob("test_pxrd_formula_lat1_*_s100.jsonl"))
    if not paths:
        paths = [
            p
            for p in sorted(eval_dir.glob("test_pxrd_formula_lat1_*.jsonl"))
            if not p.name.endswith("_summary.json")
        ]
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                recs.setdefault(row["id"], row)
    return recs


def load_energy(path: Path | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path is None or not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-eval", type=Path, required=True)
    ap.add_argument("--spec-eval", type=Path, required=True)
    ap.add_argument("--out-eval", type=Path, required=True)
    ap.add_argument("--mix-energy", type=Path, default=None)
    ap.add_argument("--spec-energy", type=Path, default=None)
    ap.add_argument("--out-energy", type=Path, default=None)
    ap.add_argument("--min-sites", type=int, default=13)
    ap.add_argument("--max-sites", type=int, default=20)
    args = ap.parse_args()

    mix = load_eval(args.mix_eval)
    spec = load_eval(args.spec_eval)
    mix_e = load_energy(args.mix_energy)
    spec_e = load_energy(args.spec_energy)

    routed: list[dict] = []
    routed_e: list[dict] = []
    n_spec = 0
    for rid, row in sorted(mix.items(), key=lambda kv: kv[0]):
        n_sites = int(row.get("target_n_sites") or 0)
        use_spec = args.min_sites <= n_sites <= args.max_sites and rid in spec
        chosen = spec[rid] if use_spec else row
        if use_spec:
            n_spec += 1
        routed.append(chosen)
        src_e = spec_e if use_spec else mix_e
        if rid in src_e:
            routed_e.append(src_e[rid])

    missing_spec = [
        rid
        for rid, row in mix.items()
        if args.min_sites <= int(row.get("target_n_sites") or 0) <= args.max_sites
        and rid not in spec
    ]
    extra_spec = [rid for rid in spec if rid not in mix]
    args.out_eval.parent.mkdir(parents=True, exist_ok=True)
    with args.out_eval.open("w") as handle:
        for row in routed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.out_energy is not None:
        args.out_energy.parent.mkdir(parents=True, exist_ok=True)
        with args.out_energy.open("w") as handle:
            for row in routed_e:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "mix": len(mix),
                "spec": len(spec),
                "routed": len(routed),
                "used_specialist": n_spec,
                "missing_specialist": missing_spec,
                "extra_specialist": extra_spec,
                "out_eval": str(args.out_eval),
                "out_energy": str(args.out_energy) if args.out_energy else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
