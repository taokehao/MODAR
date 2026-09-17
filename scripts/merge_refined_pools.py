#!/usr/bin/env python3
"""OR the match verdicts of several refinement passes over the same candidate pool.

The first pool is the base (its per-candidate ``refined`` records are kept); any
candidate matched in a later pass becomes ``structure_match=True`` with
``refined_by`` naming the pass. Candidate order is identical across passes, so
energy files keep working. ``clusters.json`` is copied when present.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", nargs="+", required=True, help="pool dirs from other refinement passes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)
    extras = [Path(p) for p in args.extra]
    out.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for p in sorted(base.glob("test_pxrd_formula_lat1_*_s100.jsonl")):
        others = []
        for ex in extras:
            q = ex / p.name
            if not q.exists():
                raise SystemExit(f"missing {q}")
            others.append({r["id"]: r for r in (json.loads(l) for l in q.open() if l.strip())})
        with (out / p.name).open("w") as f:
            for line in p.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                for k, c in enumerate(r["candidates"]):
                    for ex, o in zip(extras, others):
                        oc = o.get(r["id"], {}).get("candidates", [None] * (k + 1))[k]
                        if oc and oc.get("structure_match") and not c.get("structure_match"):
                            c["structure_match"] = True
                            c["structure_match_refined"] = True
                            c["refined_by"] = ex.name
                            n_new += 1
                r["oracle_refined"] = any(c.get("structure_match") for c in r["candidates"])
                f.write(json.dumps(r) + "\n")
    for src in [base, *extras]:
        if (src / "clusters.json").exists():
            shutil.copy(src / "clusters.json", out / "clusters.json")
            break
    print(f"wrote {out}  newly matched candidates from extra passes: {n_new}")


if __name__ == "__main__":
    main()
