#!/usr/bin/env python
"""Build the CNRS benchmark under RealPXRD-Solver's published filters.

Their CNRS set (arXiv:2603.00965, Supplementary Methods) keeps inorganic
crystalline phases with refined CIFs and primitive cells of at most 20 atoms,
and reports 126 entries. It applies no spectrum-agreement gate and no
deduplication.

Applying the same rules to our opXRD parse yields 205 entries -- a superset in
size, so the published comparison is no longer limited by our set being small.
Entry-level overlap with their 126 is unknown: opXRD carries no refinement
provenance field, so "refined CIFs" cannot be reproduced from the data, and all
1052 CNRS labels are already single-phase with `is_simulated: False`.

Raising our parse site cap from 80 to 400 grew the parsed pool from 548 to 839
without changing the primitive<=20 count, so 205 is not an artefact of that cap.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

MAX_PRIM_SITES = 20
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")

SCRATCH = PROJECT_ROOT / "data_prep/outputs/_scratch_opxrd_relaxed"
DEFAULT_OUT = PROJECT_ROOT / "data_prep/outputs/cnrs_bench205"

PROTOCOL = {
    "name": "cnrs_bench_realpxrd_filters_v1",
    "source": "opXRD CNRS labelled patterns (same raw pool as public_exp_pxrd_v1)",
    "external_reference": "arXiv:2603.00965 CNRS benchmark (126 entries)",
    "filters": {
        "inorganic": "exclude formulas containing both C and H",
        "primitive_n_sites": [2, MAX_PRIM_SITES],
        "spectrum_agreement_gate": "none (deliberately dropped)",
        "dedup": "none (deliberately dropped)",
    },
    "cell_convention": "as parsed from the opXRD label (not primitivised)",
    "caveat": (
        "Entry-level overlap with the external 126-entry set is unknown; opXRD "
        "exposes no refinement-provenance field, so their 'refined CIFs' "
        "criterion cannot be reproduced."
    ),
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_organic_or_hybrid(formula: str) -> bool:
    els = set(ELEMENT_RE.findall(formula or ""))
    return "C" in els and "H" in els


def primitive_n_sites(rec: dict) -> int | None:
    from pymatgen.core import Lattice, Structure

    try:
        st = Structure(
            Lattice(rec["lattice_matrix"]), rec["atom_types"], rec["frac_coords"]
        )
        return int(len(st.get_primitive_structure()))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", default=str(SCRATCH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pub = _load(
        Path(__file__).with_name("build_public_exp_pxrd.py"), "_pub_builder_b205"
    )
    scratch = Path(args.scratch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in (scratch / "native/test.jsonl").open()]
    kept = []
    for rec in records:
        rid = rec["id"]
        if not rid.startswith("opxrd:") or rid.startswith("opxrd:HKUST"):
            continue
        if is_organic_or_hybrid(rec["formula"]):
            continue
        n = primitive_n_sites(rec)
        if n is None or not (2 <= n <= MAX_PRIM_SITES):
            continue
        rec["experimental"]["primitive_n_sites"] = n
        kept.append(rec)

    kept.sort(key=lambda r: r["id"])
    (out_dir / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    nat = out_dir / "native/test.jsonl"
    nat.parent.mkdir(parents=True, exist_ok=True)
    with nat.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_seq, _ = pub.write_sequences(kept, out_dir / "crystal_language_v2/test.jsonl")

    cos = [r["experimental"].get("simulated_pxrd_cosine") or 0.0 for r in kept]
    prim = [r["experimental"]["primitive_n_sites"] for r in kept]
    summary = {
        "protocol": PROTOCOL["name"],
        "entries": len(kept),
        "sequences": n_seq,
        "external_target": 126,
        "cosine_below_0.40": int(sum(1 for c in cos if c < 0.40)),
        "cosine_median": sorted(cos)[len(cos) // 2] if cos else None,
        "primitive_n_sites_median": sorted(prim)[len(prim) // 2] if prim else None,
        "n_sites_as_parsed_median": sorted(int(r["n_sites"]) for r in kept)[
            len(kept) // 2
        ]
        if kept
        else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
