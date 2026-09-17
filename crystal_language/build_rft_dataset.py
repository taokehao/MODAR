#!/usr/bin/env python
"""Build a rejection-sampling (ReST / RFT) self-distillation dataset.

Stage-1 of "add RL" for the crystal LM: instead of full policy-gradient, we mine
the model's own samples that are StructureMatcher-correct and fine-tune on them.

Pipeline:
  1. ``evaluate_generation.py`` is run on a TRAIN subset (no rerank needed; the
     train split has ground truth, so ``structure_match`` already labels each
     candidate). It writes detail JSONL with ``candidates[].structure`` +
     ``structure_match`` + ``id``.
  2. This script reads those details, keeps up to ``--max-keep-per-record``
     distinct correct candidate structures per record, re-encodes each into
     canonical v2 target tokens, and emits a distilled training record that
     reuses the *source* conditioning (``cond`` + ``spectra``) with the model's
     own correct structure as the target.
  3. Records with no correct candidate optionally fall back to the ground-truth
     target (``--keep-unsolved-gt``) so coverage / anti-forgetting is preserved.

The output ``train.jsonl`` is a drop-in for ``train.py`` (same v2 schema); pair it
with the base run's ``tokenizer.json`` and original ``valid.jsonl`` for fine-tuning.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.data.sequence import site_tokens, target_tokens  # noqa: E402
from crystal_language.rerank import structure_dedup_key  # noqa: E402


def candidate_to_native(cand: dict[str, Any]) -> dict[str, Any]:
    """Map an eval candidate's stored structure to a sequence.target_tokens record."""
    st = cand["structure"]
    lat = st["lattice"]
    return {
        "atom_types": list(st["species"]),
        "frac_coords": [list(c) for c in st["frac_coords"]],
        "lattice_parameters": {
            "lengths": [lat["a"], lat["b"], lat["c"]],
            "angles": [lat["alpha"], lat["beta"], lat["gamma"]],
        },
        "space_group": int(cand.get("space_group") or 1),
        "n_sites": len(st["species"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", required=True, help="dir of evaluate_generation detail JSONL (train mining)")
    ap.add_argument("--source-train", required=True, help="source v2 train.jsonl (supplies cond + spectra)")
    ap.add_argument("--out", required=True, help="output distilled train.jsonl")
    ap.add_argument("--max-keep-per-record", type=int, default=2)
    ap.add_argument("--keep-unsolved-gt", action="store_true",
                    help="emit ground-truth target for records with no correct candidate")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    paths = sorted(p for p in globlib.glob(str(gen_dir / "*.jsonl")) if "_summary" not in p)
    if not paths:
        raise SystemExit(f"no detail JSONL found in {gen_dir}")

    kept: dict[str, list[dict[str, Any]]] = {}
    keys_by_id: dict[str, set] = {}
    seen_order: list[str] = []
    seen: set[str] = set()

    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "target_error" in row:
                    continue
                rid = row.get("id")
                if rid is None:
                    continue
                rid = str(rid)
                if rid not in seen:
                    seen.add(rid)
                    seen_order.append(rid)
                    kept[rid] = []
                    keys_by_id[rid] = set()
                for cand in row.get("candidates") or []:
                    if not cand.get("structure_match") or "structure" not in cand:
                        continue
                    if len(kept[rid]) >= args.max_keep_per_record:
                        break
                    try:
                        key = structure_dedup_key(cand["structure"])
                    except Exception:
                        continue
                    if key in keys_by_id[rid]:
                        continue
                    keys_by_id[rid].add(key)
                    kept[rid].append(candidate_to_native(cand))

    n_solved = sum(1 for rid in seen_order if kept[rid])
    print(f"mined records: {len(seen_order)}  solved(>=1 correct): {n_solved} "
          f"({100*n_solved/max(1,len(seen_order)):.1f}%)")

    # Stream source train once; keep only the conditioning for needed ids.
    src: dict[str, dict[str, Any]] = {}
    needed = set(seen_order)
    with open(args.source_train, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r.get("id"))
            if rid in needed:
                src[rid] = r
                if len(src) == len(needed):
                    break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_distill = n_gt = n_missing = 0
    with out_path.open("w", encoding="utf-8") as w:
        for rid in seen_order:
            r = src.get(rid)
            if r is None:
                n_missing += 1
                continue
            base = {k: r[k] for k in ("id", "source", "split", "cond", "spectra") if k in r}
            base["rep"] = "v2"
            if kept[rid]:
                for nat in kept[rid]:
                    rec = dict(base)
                    rec["target"] = " ".join(target_tokens(nat, rep="v2"))
                    rec["sites_target"] = " ".join(site_tokens(nat, rep="v2"))
                    rec["n_sites"] = nat["n_sites"]
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_distill += 1
            elif args.keep_unsolved_gt:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_gt += 1

    print(f"wrote {out_path}")
    print(f"  distilled (model-correct) lines: {n_distill}")
    print(f"  ground-truth fallback lines:     {n_gt}")
    print(f"  source-missing ids (skipped):    {n_missing}")
    print(f"  total lines:                     {n_distill + n_gt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
