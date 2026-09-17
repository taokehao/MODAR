#!/usr/bin/env python
"""Offline neutron-rerank ablation for the multimodal condition ladder.

The reranker score is a fixed linear combination of per-candidate consistency
terms (PXRD cosine, neutron cosine, charge balance) + a validity gate. Those
per-candidate terms are computed *once* at evaluation time and stored in the
detail JSONL (``candidates[i]["rerank"] = {pxrd_cos, nd_cos, charge_ok, valid,
sim_failed}``) together with each candidate's ``structure`` and ground-truth
``structure_match`` flag.

Changing only the *weights / policy* of the linear combination therefore needs
no re-generation: we re-score from the stored terms, re-run the exact dedup +
ordering of ``rerank.rerank_candidates``, and recompute rerank Top-k. This lets
us sweep the neutron weight ``W_ND`` and a "neutron-only-when-lattice-known"
policy in seconds.

Usage
  python crystal_language/rerank_ablation.py \
    --eval-dir crystal_language/eval_results/ladder_tierA_s2 --split test
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Reuse the exact eval-time weights / dedup so baseline reproduces merge_ladder.
from crystal_language.rerank import (  # noqa: E402
    INVALID_PENALTY,
    W_CHARGE,
    W_PXRD,
    W_ND,
    structure_dedup_key,
)

RUNGS = [
    "pxrd",
    "pxrd_element",
    "pxrd_xrf",
    "pxrd_neutron",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
]
NEUTRON_RUNGS = {"pxrd_neutron", "pxrd_neutron_xrf", "pxrd_neutron_xrf_xps"}
TOP_KS = [1, 5, 10, 20, 50]

# (label, w_nd, policy) ; policy in {always, known_only, off}
POLICIES = [
    ("baseline w_nd=0.5", 0.5, "always"),
    ("w_nd=0.25", 0.25, "always"),
    ("w_nd=0.1", 0.1, "always"),
    ("known_only w_nd=0.5", 0.5, "known_only"),
    ("neutron_rr off", 0.0, "off"),
]


def topk_any(flags: list[bool], k: int) -> bool:
    return any(flags[:k])


def load_config(eval_dir: Path, split: str, cond: str, lat: int) -> list[list[dict]]:
    """Load one (cond, lat) config; dedup records by index across shards.

    Returns a list of records, each a compact list of candidate dicts holding
    only what re-ranking needs (precomputed dedup key + consistency terms).
    """
    pattern = f"{split}_{cond}_lat{lat}_full_inject0_cd*_rr1_start*_n*_s*.jsonl"
    paths = [p for p in globlib.glob(str(eval_dir / pattern)) if "_summary" not in p]
    paths.sort(key=lambda p: int(re.search(r"start(\d+)", p).group(1)) if re.search(r"start(\d+)", p) else 0)

    seen: set = set()
    records: list[list[dict]] = []
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
                compact: list[dict] = []
                for c in row.get("candidates") or []:
                    d = c.get("rerank")
                    if "structure" in c and d:
                        compact.append({
                            "valid": True,
                            "sm": bool(c.get("structure_match")),
                            "pxrd": d.get("pxrd_cos"),
                            "nd": d.get("nd_cos"),
                            "charge": bool(d.get("charge_ok")),
                            "ok": bool(d.get("valid")) and not d.get("sim_failed"),
                            "key": structure_dedup_key(c["structure"]),
                        })
                    else:
                        compact.append({"valid": False, "sm": bool(c.get("structure_match"))})
                records.append(compact)
    return records


def reranked_flags(compact: list[dict], w_nd: float, neutron_on: bool) -> list[bool]:
    """Replicate rerank_candidates ordering with a custom neutron weight/policy.

    Returns the structure_match flags in reranked order.
    """
    valid = [i for i, c in enumerate(compact) if c["valid"]]
    invalid = [i for i, c in enumerate(compact) if not c["valid"]]

    groups: dict = {}
    for i in valid:
        groups.setdefault(compact[i]["key"], []).append(i)

    score: dict[int, float] = {}
    for members in groups.values():
        c = compact[members[0]]
        s = 0.0
        if c["pxrd"] is not None:
            s += W_PXRD * c["pxrd"]
        if neutron_on and c["nd"] is not None:
            s += w_nd * c["nd"]
        s += W_CHARGE * float(c["charge"])
        if not c["ok"]:
            s += INVALID_PENALTY
        for m in members:
            score[m] = s

    reps = [members[0] for members in groups.values()]
    # Mirror rerank.py's `(score or -inf)` idiom (a 0.0 score sinks like None).
    uniq = sorted(reps, key=lambda i: (-(score[i] or float("-inf")), i))
    dup = sorted(
        [m for members in groups.values() for m in members[1:]],
        key=lambda i: (-(score[i] or float("-inf")), i),
    )
    order = uniq + dup + invalid
    return [compact[i]["sm"] for i in order]


def eval_policy(records: list[list[dict]], w_nd: float, neutron_on: bool) -> dict[int, float]:
    n = max(1, len(records))
    hits = {k: 0 for k in TOP_KS}
    for compact in records:
        flags = reranked_flags(compact, w_nd, neutron_on)
        for k in TOP_KS:
            hits[k] += int(topk_any(flags, k))
    return {k: hits[k] / n for k in TOP_KS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)

    # Load every (cond, lat) config once.
    data: dict[tuple[str, int], list[list[dict]]] = {}
    for lat in (0, 1):
        for cond in RUNGS:
            data[(cond, lat)] = load_config(eval_dir, args.split, cond, lat)

    print(f"eval_dir={eval_dir}  split={args.split}  "
          f"(eval-time weights: W_PXRD={W_PXRD} W_ND={W_ND} W_CHARGE={W_CHARGE})")

    # Full table per policy.
    for label, w_nd, policy in POLICIES:
        print(f"\n================ policy: {label}  ({policy}) ================")
        for lat, name in ((0, "lattice_free"), (1, "known_lattice")):
            print(f"[{name}]")
            print(f"{'rung':<26}{'N':>5}{'T1':>7}{'T5':>7}{'T10':>7}{'T20':>7}{'T50':>7}")
            for cond in RUNGS:
                base_neutron = cond in NEUTRON_RUNGS
                if policy == "off":
                    neutron_on = False
                elif policy == "known_only":
                    neutron_on = base_neutron and lat == 1
                else:
                    neutron_on = base_neutron
                recs = data[(cond, lat)]
                rr = eval_policy(recs, w_nd, neutron_on)
                print(f"{cond:<26}{len(recs):>5}"
                      f"{100*rr[1]:>7.1f}{100*rr[5]:>7.1f}{100*rr[10]:>7.1f}"
                      f"{100*rr[20]:>7.1f}{100*rr[50]:>7.1f}")
            print()

    # Focused comparison: neutron rungs across policies (where the change bites).
    print("\n================ neutron-rung comparison (T1 / T5 / T10) ================")
    for lat, name in ((0, "lattice_free"), (1, "known_lattice")):
        print(f"\n[{name}]   (+xrf reference shown first; oracle = pxrd_formula)")
        ref = {}
        for cond in ("pxrd_xrf", "pxrd_formula"):
            rr = eval_policy(data[(cond, lat)], 0.5, False)
            ref[cond] = rr
            tag = "+xrf(no-neu)" if cond == "pxrd_xrf" else "ORACLE formula"
            print(f"  {tag:<22}{'':>16}T1={100*rr[1]:5.1f}  T5={100*rr[5]:5.1f}  T10={100*rr[10]:5.1f}")
        for cond in ("pxrd_neutron", "pxrd_neutron_xrf", "pxrd_neutron_xrf_xps"):
            print(f"  -- {cond}")
            for label, w_nd, policy in POLICIES:
                if policy == "off":
                    neutron_on = False
                elif policy == "known_only":
                    neutron_on = lat == 1
                else:
                    neutron_on = True
                rr = eval_policy(data[(cond, lat)], w_nd, neutron_on)
                print(f"     {label:<22}T1={100*rr[1]:5.1f}  T5={100*rr[5]:5.1f}  T10={100*rr[10]:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
