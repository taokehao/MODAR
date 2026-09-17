#!/usr/bin/env python3
"""Parse RRUFF powder DIF files into validated pymatgen structures.

A RRUFF DIF file carries the cell parameters refined against that specimen's
powder pattern, the space group, and (for most entries) the asymmetric unit of
the structure taken from the cited single-crystal determination. The atomic
coordinates are therefore literature-derived rather than refined from the same
powder measurement. This is the convention the published RRUFF benchmarks use,
and it must be stated wherever these numbers are reported.

Two things make naive parsing unsafe, so both are handled explicitly:

* RRUFF writes crystallographic site labels (``Wa`` for water oxygen, ``OH``
  for hydroxyl) that are not element symbols, and a two-letter-first fallback
  would silently turn ``Wa`` into tungsten.
* Many entries use non-standard space-group settings (``P2_1/n``, ``Pbnm``,
  ``I2``) whose operations differ from the standard setting, so relabelling to
  the standard symbol would move every atom.

Every reconstructed structure is therefore checked against the reflection list
the DIF file itself carries: that list was computed from the true structure, so
reproducing it is direct evidence that the reconstruction is faithful.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Real ``SPACE GROUP:`` lines only. Files also contain
# "ALTERNATE SETTING FOR SPACE GROUP: ..." whose symbol is often truncated.
SPG_RE = re.compile(r"(?<!FOR )SPACE GROUP:\s*([^\r\n]+)")
# 182 entries give only the alternate setting. That symbol is the one the
# coordinates are expressed in, so it is usable, but it is sometimes truncated
# in the file -- the reflection-list check below is what makes accepting it safe.
ALT_SPG_RE = re.compile(r"ALTERNATE SETTING FOR SPACE GROUP:\s*([^\r\n]+)")
CELL_RE = re.compile(r"CELL PARAMETERS:\s*([^\r\n]+)")
WAVELENGTH_RE = re.compile(r"X-RAY WAVELENGTH:\s*([0-9.]+)")
ATOM_HEADER_RE = re.compile(r"^\s*ATOM\s+X\s+Y\s+Z", re.IGNORECASE)
ATOM_ROW_RE = re.compile(
    r"^\s*(\S+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)"
    r"(?:\s+(-?[0-9.]+))?"
)
REFLECTION_RE = re.compile(
    r"^\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$"
)
RRUFF_ID_RE = re.compile(r"__([A-Z]\d+(?:-\d+)?)__")

# RRUFF site labels that are not element symbols. Resolved explicitly rather
# than by a two-letter-then-one-letter fallback, which would read water oxygen
# ("Wa") as tungsten and sulfide-layer sulfur ("Sl") as an unknown element.
LABEL_TO_ELEMENT = {
    "WA": "O",   # water oxygen
    "OW": "O",   # water oxygen
    "OH": "O",   # hydroxyl oxygen
    "OA": "O",
    "OB": "O",
    "OC": "O",
    "OD": "O",
    "OE": "O",
    "OL": "O",
    "OO": "O",
    "OP": "O",
    "SL": "S",
}

DUP_TOL = 1e-3


@dataclass
class DifRecord:
    path: Path
    rruff_id: str
    mineral: str
    spacegroup: str
    wavelength: float | None
    cell: tuple[float, float, float, float, float, float]
    atoms: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    reflections: list[tuple[float, float]] = field(default_factory=list)
    reference: str = ""

    @property
    def has_atoms(self) -> bool:
        return bool(self.atoms)


def rruff_id_from_name(path: Path) -> str | None:
    m = RRUFF_ID_RE.search(path.name)
    return m.group(1) if m else None


def resolve_element(label: str) -> str | None:
    from pymatgen.core.periodic_table import Element

    token = re.sub(r"[0-9+\-*'\"]+$", "", label).strip()
    if not token:
        return None
    mapped = LABEL_TO_ELEMENT.get(token.upper())
    if mapped:
        return mapped
    for cand in (token.capitalize(), token[:1].upper()):
        try:
            Element(cand)
            return cand
        except Exception:
            continue
    return None


def parse_dif(path: Path) -> DifRecord | None:
    text = path.read_text(errors="replace")
    cell_m = CELL_RE.search(text)
    spg_m = SPG_RE.search(text) or ALT_SPG_RE.search(text)
    if not cell_m or not spg_m:
        return None
    # The cell line is fixed-width, so a long value can run into its neighbour
    # ("8.1910201.6500"). Such a record is unrecoverable, not merely unparsed.
    try:
        nums = [float(v) for v in cell_m.group(1).split()[:6]]
    except ValueError:
        return None
    if len(nums) != 6:
        return None
    wl_m = WAVELENGTH_RE.search(text)
    lines = text.splitlines()

    atoms: list[tuple[str, float, float, float, float]] = []
    in_atoms = False
    for line in lines:
        if ATOM_HEADER_RE.match(line):
            in_atoms = True
            continue
        if not in_atoms:
            continue
        if not line.strip():
            if atoms:
                break
            continue
        m = ATOM_ROW_RE.match(line)
        if not m:
            break
        label, x, y, z, occ = m.groups()
        el = resolve_element(label)
        if el is None:
            return None
        atoms.append((el, float(x), float(y), float(z), float(occ) if occ else 1.0))

    reflections: list[tuple[float, float]] = []
    for line in lines:
        m = REFLECTION_RE.match(line)
        if m:
            tt, inten = float(m.group(1)), float(m.group(2))
            if inten > 0:
                reflections.append((tt, inten))

    return DifRecord(
        path=path,
        rruff_id=rruff_id_from_name(path) or path.stem,
        mineral=lines[0].strip() if lines else "",
        spacegroup=spg_m.group(1).strip(),
        wavelength=float(wl_m.group(1)) if wl_m else None,
        cell=tuple(nums),  # type: ignore[arg-type]
        atoms=atoms,
        reflections=reflections,
        reference=" ".join(lines[3:6]).strip() if len(lines) > 5 else "",
    )


def spacegroup_ops(symbol: str):
    """Resolve a RRUFF space-group symbol to gemmi symmetry operations."""
    import gemmi

    s = symbol.strip()
    candidates = [s, s.replace("_", "")]
    # Monoclinic non-standard settings are written unique-axis-b in RRUFF; the
    # explicit three-part form is what gemmi's table indexes them under.
    m = re.match(r"^([PABCIF])2_?1/([abcn])$", s)
    if m:
        lattice, glide = m.group(1), m.group(2)
        candidates.append(f"{lattice} 1 21/{glide} 1")  # unique axis b
        candidates.append(f"{lattice} 1 1 21/{glide}")  # unique axis c
    for cand in candidates:
        try:
            sg = gemmi.SpaceGroup(cand)
        except Exception:
            continue
        return sg, cand
    return None, None


def structure_from_dif(rec: DifRecord):
    """Expand the asymmetric unit by the space group into a full structure."""
    from pymatgen.core import Lattice, Structure

    if not rec.has_atoms:
        return None, "no_atoms"
    a, b, c, alpha, beta, gamma = rec.cell
    if min(a, b, c) <= 0 or not all(0 < ang < 180 for ang in (alpha, beta, gamma)):
        return None, "bad_cell"
    sg, used = spacegroup_ops(rec.spacegroup)
    if sg is None:
        return None, f"unknown_spacegroup:{rec.spacegroup}"

    ops = sg.operations()
    species: list[str] = []
    coords: list[list[float]] = []
    for el, x, y, z, occ in rec.atoms:
        if occ < 0.5:
            # Partially occupied sites cannot be represented by the ordered
            # tokenizer; dropping the minor component keeps the majority model.
            continue
        seen: list[np.ndarray] = []
        for op in ops:
            fx, fy, fz = op.apply_to_xyz([x, y, z])
            v = np.array([fx % 1.0, fy % 1.0, fz % 1.0])
            if any(
                np.all(np.abs((v - u + 0.5) % 1.0 - 0.5) < DUP_TOL) for u in seen
            ):
                continue
            seen.append(v)
            species.append(el)
            coords.append(v.tolist())
    if not species:
        return None, "no_sites"
    try:
        st = Structure(
            Lattice.from_parameters(a, b, c, alpha, beta, gamma),
            species,
            coords,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"structure_error:{type(exc).__name__}"
    return st, used


def pattern_cosine(
    structure, reflections: list[tuple[float, float]], wavelength: float
) -> float | None:
    """Cosine between PXRD simulated from ``structure`` and the DIF reflections.

    The DIF reflection list was computed from the true structure, so agreement
    is direct evidence that the symmetry expansion and element assignment are
    right. Peaks are rasterized on a shared grid to tolerate small position
    differences.
    """
    if not reflections:
        return None
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    lo, hi, step, sigma = 5.0, 90.0, 0.05, 0.15
    n = int(round((hi - lo) / step)) + 1
    grid = np.arange(n) * step + lo

    def raster(xs, ys):
        vec = np.zeros(n)
        for x, y in zip(xs, ys):
            if y <= 0 or x < lo or x > hi:
                continue
            vec += y * np.exp(-((grid - x) ** 2) / (2 * sigma * sigma))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    try:
        pat = XRDCalculator(wavelength=wavelength).get_pattern(
            structure, scaled=True, two_theta_range=(lo, hi)
        )
    except Exception:
        return None
    sim = raster(pat.x, pat.y)
    ref = raster([t for t, _ in reflections], [i for _, i in reflections])
    return float(np.dot(sim, ref))


def audit(dif_dir: Path, limit: int | None = None, cos_min: float = 0.90) -> dict[str, Any]:
    paths = sorted(dif_dir.glob("*.txt"))
    if limit:
        paths = paths[:limit]
    stats = {
        "files": len(paths),
        "parsed": 0,
        "with_atoms": 0,
        "structure_ok": 0,
        "validated": 0,
        "cos_min": cos_min,
    }
    reasons: dict[str, int] = {}
    cosines: list[float] = []
    n_sites: list[int] = []
    for p in paths:
        rec = parse_dif(p)
        if rec is None:
            reasons["parse_failed"] = reasons.get("parse_failed", 0) + 1
            continue
        stats["parsed"] += 1
        if not rec.has_atoms:
            reasons["no_atoms"] = reasons.get("no_atoms", 0) + 1
            continue
        stats["with_atoms"] += 1
        st, info = structure_from_dif(rec)
        if st is None:
            reasons[str(info)] = reasons.get(str(info), 0) + 1
            continue
        stats["structure_ok"] += 1
        cos = pattern_cosine(st, rec.reflections, rec.wavelength or 1.541838)
        if cos is None:
            reasons["no_reflections"] = reasons.get("no_reflections", 0) + 1
            continue
        cosines.append(cos)
        if cos >= cos_min:
            stats["validated"] += 1
            try:
                n_sites.append(len(st.get_primitive_structure()))
            except Exception:
                n_sites.append(len(st))
        else:
            reasons["low_pattern_cosine"] = reasons.get("low_pattern_cosine", 0) + 1
    stats["reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:15])
    if cosines:
        arr = np.array(cosines)
        stats["pattern_cosine"] = {
            "median": float(np.median(arr)),
            "ge_0.99": int((arr >= 0.99).sum()),
            "ge_0.95": int((arr >= 0.95).sum()),
            "ge_0.90": int((arr >= 0.90).sum()),
            "lt_0.50": int((arr < 0.50).sum()),
        }
    if n_sites:
        arr = np.array(n_sites)
        stats["primitive_n_sites"] = {
            "median": float(np.median(arr)),
            "le_20": int((arr <= 20).sum()),
            "le_80": int((arr <= 80).sum()),
        }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dif-dir", default="data_prep/outputs/rruff_exp_pxrd/raw/dif")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cos-min", type=float, default=0.90)
    args = ap.parse_args()
    print(json.dumps(audit(Path(args.dif_dir), args.limit, args.cos_min), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DifRecord",
    "parse_dif",
    "pattern_cosine",
    "resolve_element",
    "rruff_id_from_name",
    "structure_from_dif",
]
