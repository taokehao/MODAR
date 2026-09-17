#!/usr/bin/env python
"""Build a frozen public experimental-PXRD eval set (COD + optional opXRD).

This hold-out is meant to be reported *before* the collaborator 10-sample
case study. Filters are written to ``protocol.json`` before any parsing, and
they do not depend on model ranks.

Open sources only (ICDD / PDF-5+ / CSD / ICSD powder are not used):

* Crystallography Open Database — powder diffraction, has Fobs, not neutron.
  Observed pdCIF / HKL intensities + deposited CIF.
* opXRD (Hollarek et al.; Zenodo 10.5281/zenodo.14254270) if ``--opxrd-dir``
  or ``--download-opxrd`` is given. Only JSON patterns with a full crystal
  structure (lattice + basis) are scored. CC-BY-4.0.

RRUFF is not scored (no atomic coordinates). HTEM-DB is not scored (no
structure).

PXRD is the experimental pattern, Q-aligned onto the training grid
(λ=1.54184 Å, 5–80°). Neutron / XRF / XPS are simulated from the CIF.

Usage::

  python data_prep/official_multisource/scripts/build_public_exp_pxrd.py
  python data_prep/official_multisource/scripts/build_public_exp_pxrd.py --download-opxrd
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

NPD_PATH = Path(__file__).with_name("build_cod_experimental_npd.py")
COLLAB_PATH = Path(__file__).with_name("build_collaborator_exp_pxrd.py")

GRID_MIN = 5.0
GRID_MAX = 80.0
REF_WAVELENGTH = 1.54184
MAX_SITES = 80
MIN_PEAKS = 8
COSINE_MIN = 0.40
OPXRD_URL = "https://zenodo.org/records/15298026/files/opxrd.zip?download=1"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "public_exp_pxrd"
TOKENIZER_PATH = (
    PROJECT_ROOT / "crystal_language" / "runs" / "tierA_s2_noisy_exp" / "tokenizer.json"
)

PROTOCOL = {
    "name": "public_exp_pxrd_v1",
    "frozen_before_eval": True,
    "selection": "pre-registered filters only; no model-based keep/drop",
    "sources": {
        "cod": {
            "query": "method=powder diffraction AND flags has Fobs AND not neutron",
            "pattern": "observed pdCIF/HKL only; never _pd_calc_*",
            "structure": "deposited CIF, majority species per site, D→H",
        },
        "opxrd": {
            "record": "10.5281/zenodo.14254270",
            "keep": "JSON patterns whose label has lattice AND atomic basis",
            "skip": "composition-only, lattice-only, or no primary basis",
            "download": "labeled CNRS + HKUST-A only (Hollarek Table 1 atom coords); not the 1.4 GB unlabeled dump",
        },
        "excluded": [
            "ICDD/PDF-5+",
            "CSD powder",
            "ICSD powder",
            "RRUFF (no atomic coordinates)",
            "HTEM-DB (no structure)",
        ],
    },
    "grid": {"two_theta": [GRID_MIN, GRID_MAX], "step": 0.05, "wavelength_A": REF_WAVELENGTH},
    "scored_if": {
        "n_sites": [2, MAX_SITES],
        "n_pxrd_bins": f">={MIN_PEAKS}",
        "simulated_pxrd_cosine": f">={COSINE_MIN}",
        "occupancy_collapsed": False,
        "tokenizer_legal_elements": True,
        "single_phase_structure": True,
    },
    "dedup": "reduced_formula + space_group; keep highest simulated_pxrd_cosine",
    "modalities": {
        "pxrd": "experimental, Q-aligned to training λ",
        "neutron": "simulated from CIF at training λ (not experimental ND)",
        "xrf": "occupancy-weighted CIF composition",
        "xps": "heuristic valences from CIF",
    },
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


npd = _load_module(NPD_PATH, "cod_npd_helpers")
collab = _load_module(COLLAB_PATH, "collab_pxrd_helpers")


def _alias(mod, dest: str, *sources: str) -> None:
    if hasattr(mod, dest):
        return
    for name in sources:
        if hasattr(mod, name):
            setattr(mod, dest, getattr(mod, name))
            return


# dest = name used in this file; sources = names in the helper modules.
_alias(npd, "COD_MYSQL", "COD_MYSQL")
_alias(npd, "COD_HTTP", "COD_HTTP")
_alias(npd, "INTENSITY_TAGS", "INTENSITY_TAGS")
_alias(npd, "parse_cif_blocks", "parse_cif_blocks")
_alias(npd, "_block_wavelength", "_block_wavelength")
_alias(npd, "_is_neutron_block", "_is_neutron_block")
_alias(npd, "_pattern_from_loop", "_pattern_from_loop")
_alias(npd, "site_symbol", "site_symbol")
_alias(npd, "remap_two_theta_to_train", "remap_two_theta_to_train")
_alias(npd, "clip_and_peaks", "clip_and_peaks")
_alias(npd, "_finite", "_finite")
_alias(npd, "_parse_wavelength", "_parse_wavelength")
_alias(npd, "structure_from_cif", "structure_from_cif")
_alias(npd, "_normalize_symbol", "_normalize_symbol")
_alias(collab, "encode_profile", "encode_profile")
_alias(collab, "cosine_on_grid", "cosine_on_grid")
_alias(collab, "xrf_from_atoms", "xrf_from_atoms")

_orig_parse = npd.parse_cif_blocks


def _parse_cif_blocks(text: str):
    blocks = _orig_parse(text)
    for block in blocks:
        if "items" in block and "items" not in block:
            block["items"] = block["items"]
        if "items" in block and "items" not in block:
            block["items"] = block["items"]
        if "loops" in block and "loops" not in block:
            block["loops"] = block["loops"]
        if "loops" in block and "loops" not in block:
            block["loops"] = block["loops"]
    return blocks


npd.parse_cif_blocks = _parse_cif_blocks

_orig_clip = npd.clip_and_peaks


def _clip_and_peaks(xs, ys, dense_profile: bool = False, **kwargs):
    dense = kwargs.pop("dense_profile", dense_profile)
    try:
        return _orig_clip(xs, ys, dense_profile=dense, **kwargs)
    except TypeError:
        return _orig_clip(xs, ys, dense_profile=dense, **kwargs)


npd.clip_and_peaks = _clip_and_peaks

_orig_download = npd.download


def _download(url, dest, retries: int = 4, timeout: int = 60, **kwargs):
    retries = int(kwargs.pop("retries", retries))
    timeout = int(kwargs.pop("timeout", timeout))
    return _orig_download(url, dest, retries=retries, timeout=timeout)


npd.download = _download


def load_tokenizer_elements(path: Path) -> set[str]:
    if not path.exists():
        return set()
    vocab = json.loads(path.read_text(encoding="utf-8")).get("vocab") or []
    return {tok for tok in vocab if re.fullmatch(r"[A-Z][a-z]?", str(tok))}


def query_cod_catalog() -> list[dict[str, str]]:
    sqls = [
        """
SELECT file, formula, calcformula, nel, sg, sgNumber, wavelength, radiation,
       radType, flags, journal, year, doi, title
FROM data
WHERE method='powder diffraction'
  AND flags LIKE '%has Fobs%'
  AND NOT (IFNULL(radiation,'') LIKE '%neutron%' OR IFNULL(radType,'') LIKE '%neutron%')
ORDER BY year DESC, file;
""",
        """
SELECT file, formula, calcformula, nel, sg, sgNumber, wavelength, radiation,
       radType, flags, journal, year, doi, title
FROM data
WHERE method='powder diffraction'
  AND flags LIKE '%has Fobs%'
  AND NOT (IFNULL(radiation,'') LIKE '%neutron%' OR IFNULL(radType,'') LIKE '%neutron%')
ORDER BY year DESC, file;
""",
    ]
    last_err: Exception | None = None
    for sql in sqls:
        proc = subprocess.run(
            [*npd.COD_MYSQL, "-e", sql, "cod"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            last_err = RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "COD query failed")
            continue
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            last_err = RuntimeError("COD PXRD query returned no rows")
            continue
        header = lines[0].split("\t")
        rows = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(dict(zip(header, parts)))
        return rows
    raise last_err or RuntimeError("COD PXRD query failed")


def extract_observed_pxrd(
    texts: list[str],
    catalog_wavelength: float | None,
) -> dict[str, Any] | None:
    orig_tags = npd.INTENSITY_TAGS
    npd.INTENSITY_TAGS = tuple(tag for tag in orig_tags if "calc" not in tag)
    try:
        blocks: list[dict[str, Any]] = []
        for text in texts:
            blocks.extend(npd.parse_cif_blocks(text))
        if not blocks:
            return None
        wavelength = catalog_wavelength
        if wavelength is None:
            for block in blocks:
                wavelength = npd._block_wavelength(block, wavelength)
        if wavelength is None:
            wavelength = REF_WAVELENGTH
        merged: dict[str, str] = {}
        for block in blocks:
            merged.update(block.get("items") or {})
        best: dict[str, Any] | None = None
        for block in blocks:
            if npd._is_neutron_block(block):
                continue
            items = {**merged, **(block.get("items") or {})}
            for loop in block.get("loops") or []:
                try:
                    parsed = npd._pattern_from_loop(loop, wavelength, items)
                except Exception:
                    continue
                if parsed is None:
                    continue
                xs, ys, kind = parsed
                if "calc" in kind:
                    continue
                score = int("profile" in kind) * 1_000_000 + len(xs)
                if best is None or score > best["score"]:
                    best = {
                        "x": xs,
                        "y": ys,
                        "kind": kind,
                        "wavelength": wavelength,
                        "block": block.get("name"),
                        "score": score,
                        "n_raw": int(len(xs)),
                    }
        return best
    finally:
        npd.INTENSITY_TAGS = orig_tags


def occupancy_collapsed(row: dict[str, str], structure: Any) -> list[str]:
    model_elems = {npd.site_symbol(site) for site in structure}
    catalog_elems: set[str] = set()
    for raw in (row.get("calcformula"), row.get("formula")):
        if not raw:
            continue
        catalog_elems.update(re.findall(r"[A-Z][a-z]?", raw.replace("D", "H")))
    return sorted(el for el in catalog_elems if el not in model_elems and el not in {"D"})


def tokenizer_illegal(atom_types: list[str], legal: set[str]) -> list[str]:
    if not legal:
        return []
    return sorted({sym for sym in atom_types if sym not in legal})


def encode_pattern(pattern: dict[str, Any]) -> tuple[list[float], list[float], int, bool]:
    xs = np.asarray(pattern["x"], dtype=float)
    ys = np.asarray(pattern["y"], dtype=float)
    wavelength = float(pattern.get("wavelength") or REF_WAVELENGTH)
    xs, ys, q_aligned = npd.remap_two_theta_to_train(xs, ys, wavelength, REF_WAVELENGTH)
    kind = str(pattern.get("kind") or "")
    dense = "profile" in kind or xs.size > 200
    if dense:
        x_clip, _, peak_x, peak_y = collab.encode_profile(xs, ys)
    else:
        x_clip, _, peak_x, peak_y = npd.clip_and_peaks(xs, ys, dense_profile=False)
    return peak_x, peak_y, len(x_clip), q_aligned


def valence_fields(valence: dict[str, Any]) -> dict[str, Any]:
    assignments = valence.get("valence_assignments") or valence.get("valence_assignments") or {}
    source = valence.get("valence_source") or valence.get("valence_source")
    assigned = valence.get("valence_assigned")
    if assigned is None:
        assigned = valence.get("valence_assigned")
    return {
        "valence_assignments": assignments,
        "valence_source": source,
        "valence_assigned": bool(assigned),
    }


def to_native_record(
    *,
    sample_id: str,
    source: str,
    source_id: str,
    row: dict[str, str],
    structure: Any,
    peak_x: list[float],
    peak_y: list[float],
    pattern_meta: dict[str, Any],
    n_profile: int,
    q_aligned: bool,
    disordered: bool,
) -> dict[str, Any]:
    from pymatgen.core import Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from src.multimodal_utils import compute_neutron_pattern, compute_valence
    from src.structure_utils import compute_pxrd

    atom_types = [npd.site_symbol(site) for site in structure]
    frac_coords = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    lattice = Lattice(lattice_matrix)
    sim_pxrd_x, sim_pxrd_y = compute_pxrd(structure, GRID_MIN, GRID_MAX)
    sim_nd = compute_neutron_pattern(structure, GRID_MIN, GRID_MAX, wavelength=REF_WAVELENGTH)
    valence = compute_valence(structure)
    try:
        space_group = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
    except Exception:
        space_group = int(npd._finite(row.get("sgNumber")) or 0)

    sim_pxrd_y = (sim_pxrd_y / max(float(np.max(sim_pxrd_y)), 1e-12)).tolist()
    sim_pxrd_x = sim_pxrd_x.tolist()
    sim_nd_x = sim_nd[0].tolist() if sim_nd is not None else []
    sim_nd_y = (sim_nd[1] / max(float(np.max(sim_nd[1])), 1e-12)).tolist() if sim_nd is not None else []
    pxrd_cos = collab.cosine_on_grid(peak_x, peak_y, sim_pxrd_x, sim_pxrd_y)
    xrf = collab.xrf_from_atoms(atom_types)
    vfields = valence_fields(valence)

    return {
        "id": sample_id,
        "source": source,
        "source_id": str(source_id),
        "split": "test",
        "atom_types": atom_types,
        "atom_types": atom_types,
        "frac_coords": frac_coords.tolist(),
        "lattice_matrix": lattice_matrix.tolist(),
        "lattice_parameters": {
            "lengths": [float(v) for v in lattice.abc],
            "angles": [float(v) for v in lattice.angles],
        },
        "formula": structure.composition.reduced_formula,
        "n_sites": int(len(structure)),
        "space_group": space_group,
        "conditions": {
            "pxrd_x": [round(float(v), 3) for v in peak_x],
            "pxrd_y": [round(float(v), 3) for v in peak_y],
            "neutron_x": [round(float(v), 3) for v in sim_nd_x],
            "neutron_y": [round(float(v), 3) for v in sim_nd_y],
            "xrf_mass_frac": xrf,
            **vfields,
        },
        "properties": {
            "formation_energy_per_atom": None,
            "e_above_hull": None,
            "band_gap": None,
        },
        "experimental": {
            "doi": row.get("doi") or None,
            "journal": row.get("journal") or None,
            "year": int(npd._finite(row.get("year")) or 0) or None,
            "title": row.get("title") or None,
            "catalog_formula": row.get("formula") or None,
            "wavelength_A": pattern_meta.get("wavelength"),
            "pattern_kind": pattern_meta.get("kind"),
            "n_profile_points": n_profile,
            "n_pxrd_bins": len(peak_x),
            "q_aligned": bool(q_aligned),
            "disordered_cif": bool(disordered),
            "simulated_pxrd_cosine": pxrd_cos,
        },
    }


def process_cod_entry(
    row: dict[str, str],
    raw_dir: Path,
    legal_elements: set[str],
    max_sites: int,
) -> dict[str, Any]:
    cod_id = str(row["file"]).zfill(7)
    status: dict[str, Any] = {
        "id": f"cod:{cod_id}",
        "source": "cod",
        "status": "failed",
        "reason": None,
        "formula": row.get("formula"),
        "year": row.get("year"),
        "doi": row.get("doi"),
    }
    try:
        return _process_cod_entry_inner(row, raw_dir, legal_elements, max_sites, status, cod_id)
    except Exception as exc:
        status["reason"] = f"unhandled:{type(exc).__name__}"
        return status


def _process_cod_entry_inner(
    row: dict[str, str],
    raw_dir: Path,
    legal_elements: set[str],
    max_sites: int,
    status: dict[str, Any],
    cod_id: str,
) -> dict[str, Any]:
    cif_path = raw_dir / f"{cod_id}.cif"
    hkl_path = raw_dir / f"{cod_id}.hkl"
    if not npd.download(f"{npd.COD_HTTP}/{cod_id}.cif", cif_path):
        status["reason"] = "cif_download_failed"
        return status
    npd.download(f"{npd.COD_HTTP}/{cod_id}.hkl", hkl_path)
    texts = [cif_path.read_text(encoding="utf-8", errors="replace")]
    if hkl_path.exists() and hkl_path.stat().st_size > 0:
        texts.append(hkl_path.read_text(encoding="utf-8", errors="replace"))
    catalog_wavelength = npd._parse_wavelength(row.get("wavelength"))
    pattern = extract_observed_pxrd(texts, catalog_wavelength)
    if pattern is None:
        status["reason"] = "no_observed_pxrd"
        return status
    peak_x, peak_y, n_profile, q_aligned = encode_pattern(pattern)
    if len(peak_x) < MIN_PEAKS:
        status["reason"] = f"too_few_pxrd_bins:{len(peak_x)}"
        return status
    try:
        structure = npd.structure_from_cif(texts[0])
    except Exception as exc:
        status["reason"] = f"structure_parse_failed:{type(exc).__name__}"
        return status
    n_sites = len(structure)
    if n_sites < 2:
        status["reason"] = "too_few_sites"
        return status
    if n_sites > max_sites:
        status["reason"] = f"too_many_sites:{n_sites}"
        status["n_sites"] = n_sites
        return status
    missing = occupancy_collapsed(row, structure)
    if missing:
        status["reason"] = f"occupancy_collapsed:{missing}"
        status["n_sites"] = n_sites
        return status
    atom_types = [npd.site_symbol(site) for site in structure]
    illegal = tokenizer_illegal(atom_types, legal_elements)
    if illegal:
        status["reason"] = f"tokenizer_illegal:{illegal}"
        return status
    disordered = any(not getattr(site, "is_ordered", True) for site in structure)
    native = to_native_record(
        sample_id=f"cod:{cod_id}",
        source="cod_experimental_pxrd",
        source_id=cod_id,
        row=row,
        structure=structure,
        peak_x=peak_x,
        peak_y=peak_y,
        pattern_meta=pattern,
        n_profile=n_profile,
        q_aligned=q_aligned,
        disordered=disordered,
    )
    cosine = native["experimental"].get("simulated_pxrd_cosine")
    status.update(
        {
            "status": "ok",
            "formula": native["formula"],
            "n_sites": native["n_sites"],
            "space_group": native["space_group"],
            "n_pxrd_bins": len(peak_x),
            "pattern_kind": pattern["kind"],
            "wavelength_A": pattern["wavelength"],
            "q_aligned": q_aligned,
            "disordered_cif": disordered,
            "simulated_pxrd_cosine": cosine,
            "native": native,
        }
    )
    return status


def _maybe_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _walk_json(obj: Any) -> Any:
    obj = _maybe_json(obj)
    if isinstance(obj, dict):
        return {str(k): _walk_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_json(v) for v in obj]
    return obj


def _as_float_list(value: Any) -> list[float] | None:
    value = _maybe_json(value)
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for item in value:
        number = npd._finite(item)
        if number is None:
            return None
        out.append(float(number))
    return out


def _lattice_from_obj(obj: Any):
    from pymatgen.core import Lattice

    obj = _walk_json(obj)
    if obj is None:
        return None
    if isinstance(obj, str) and obj.startswith("(") and obj.endswith(")"):
        parts = [npd._finite(p) for p in obj[1:-1].split(",")]
        if len(parts) == 6 and all(p is not None for p in parts):
            return Lattice.from_parameters(*[float(p) for p in parts])
        return None
    if isinstance(obj, dict):
        if "matrix" in obj:
            try:
                return Lattice(obj["matrix"])
            except Exception:
                return None
        params = obj.get("parameters") or obj.get("abc")
        if params is not None and len(params) == 6:
            try:
                return Lattice.from_parameters(*[float(x) for x in params])
            except Exception:
                return None
        lengths = obj.get("lengths") or [obj.get("a"), obj.get("b"), obj.get("c")]
        angles = obj.get("angles") or [obj.get("alpha"), obj.get("beta"), obj.get("gamma")]
        if lengths and angles and None not in lengths and None not in angles:
            try:
                return Lattice.from_parameters(
                    float(lengths[0]),
                    float(lengths[1]),
                    float(lengths[2]),
                    float(angles[0]),
                    float(angles[1]),
                    float(angles[2]),
                )
            except Exception:
                return None
    return None


def _structure_from_opxrd_label(label: Any):
    from pymatgen.core import Structure

    label = _walk_json(label)
    if not isinstance(label, dict):
        return None
    phases = label.get("phases") or label.get("phase_list") or []
    if isinstance(phases, dict):
        phases = [phases]
    if not phases and ("lattice" in label or "basis" in label):
        phases = [label]
    if not phases:
        return None
    phase = phases[0]
    if not isinstance(phase, dict):
        return None
    lattice = _lattice_from_obj(phase.get("lattice"))
    basis = _walk_json(phase.get("basis") or phase.get("sites") or phase.get("atom_sites"))
    if lattice is None or not isinstance(basis, list) or len(basis) < 2:
        return None
    species: list[str] = []
    coords: list[list[float]] = []
    for site in basis:
        site = _walk_json(site)
        if not isinstance(site, dict):
            continue
        raw_sym = (
            site.get("symbol")
            or site.get("species_str")
            or site.get("species")
            or site.get("element")
            or site.get("atom")
            or ""
        )
        if isinstance(raw_sym, dict):
            raw_sym = raw_sym.get("symbol") or raw_sym.get("element") or ""
        symbol = npd._normalize_symbol(re.sub(r"[^A-Za-z].*$", "", str(raw_sym)))
        if not symbol:
            continue
        abc = site.get("abc") if isinstance(site.get("abc"), list) else None
        x = npd._finite(site.get("x") if "x" in site else (abc[0] if abc else None))
        y = npd._finite(site.get("y") if "y" in site else (abc[1] if abc and len(abc) > 1 else site.get("frac_y")))
        z = npd._finite(site.get("z") if "z" in site else (abc[2] if abc and len(abc) > 2 else site.get("frac_z")))
        if x is None or y is None or z is None:
            xyz = site.get("xyz") or site.get("frac_coords")
            if isinstance(xyz, (list, tuple)) and len(xyz) == 3:
                x, y, z = npd._finite(xyz[0]), npd._finite(xyz[1]), npd._finite(xyz[2])
        if x is None or y is None or z is None:
            continue
        occ = npd._finite(site.get("occupancy") or site.get("occ")) or 1.0
        if occ < 0.20:
            continue
        species.append(symbol)
        coords.append([float(x) % 1.0, float(y) % 1.0, float(z) % 1.0])
    if len(species) < 2:
        return None
    return Structure(lattice, species, coords, coords_are_cartesian=False)


def process_opxrd_file(path: Path, legal_elements: set[str], max_sites: int) -> dict[str, Any]:
    rel = path.as_posix()
    rel_id = path.as_posix()
    for marker in ("/raw/opxrd/", "/opxrd/"):
        if marker in rel_id:
            rel_id = rel_id.split(marker, 1)[1]
            break
    else:
        rel_id = f"{path.parent.name}/{path.name}"
    rel_id = rel_id.replace("\\", "/").removesuffix(".json")
    status: dict[str, Any] = {
        "id": f"opxrd:{rel_id}",
        "source": "opxrd",
        "status": "failed",
        "reason": None,
        "path": str(path),
        "doi": "10.5281/zenodo.14254270",
    }
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        status["reason"] = "json_parse_failed"
        return status
    data = _walk_json(raw)
    if not isinstance(data, dict):
        status["reason"] = "json_not_object"
        return status
    xs = _as_float_list(
        data.get("two_theta_values")
        or data.get("two_theta")
        or data.get("x")
        or data.get("angles")
    )
    ys = _as_float_list(data.get("intensities") or data.get("y") or data.get("intensity"))
    if xs is None or ys is None or len(xs) != len(ys) or len(xs) < 50:
        status["reason"] = "no_pattern"
        return status
    label = data.get("label") or data.get("powder_experiment") or data.get("metadata")
    structure = _structure_from_opxrd_label(label)
    if structure is None:
        status["reason"] = "no_full_structure"
        return status
    wavelength = REF_WAVELENGTH
    blobs = [data]
    walked_label = _walk_json(label)
    if isinstance(walked_label, dict):
        blobs.append(walked_label)
        for phase in walked_label.get("phases") or []:
            if isinstance(phase, dict):
                blobs.append(phase)
    for blob in blobs:
        if not isinstance(blob, dict):
            continue
        for key in ("primary_wavelength", "wavelength", "wavelength_A"):
            parsed = npd._parse_wavelength(blob.get(key))
            if parsed is not None:
                wavelength = parsed
        xray = blob.get("xray_info") or blob.get("xray") or {}
        xray = _walk_json(xray)
        if isinstance(xray, dict):
            parsed = npd._parse_wavelength(
                xray.get("primary_wavelength")
                or xray.get("wavelength")
                or xray.get("anode_wavelength")
            )
            if parsed is not None:
                wavelength = parsed
    pattern = {"x": np.asarray(xs), "y": np.asarray(ys), "kind": "profile", "wavelength": wavelength}
    peak_x, peak_y, n_profile, q_aligned = encode_pattern(pattern)
    if len(peak_x) < MIN_PEAKS:
        status["reason"] = f"too_few_pxrd_bins:{len(peak_x)}"
        return status
    n_sites = len(structure)
    if n_sites < 2:
        status["reason"] = "too_few_sites"
        return status
    if n_sites > max_sites:
        status["reason"] = f"too_many_sites:{n_sites}"
        status["n_sites"] = n_sites
        return status
    atom_types = [npd.site_symbol(site) for site in structure]
    illegal = tokenizer_illegal(atom_types, legal_elements)
    if illegal:
        status["reason"] = f"tokenizer_illegal:{illegal}"
        return status
    row = {
        "formula": structure.composition.reduced_formula,
        "doi": "10.5281/zenodo.14254270",
        "title": rel,
        "year": "2025",
        "journal": "opXRD",
        "sgNumber": "",
    }
    disordered = any(not getattr(site, "is_ordered", True) for site in structure)
    native = to_native_record(
        sample_id=f"opxrd:{rel_id}",
        source="opxrd_experimental_pxrd",
        source_id=rel_id,
        row=row,
        structure=structure,
        peak_x=peak_x,
        peak_y=peak_y,
        pattern_meta=pattern,
        n_profile=n_profile,
        q_aligned=q_aligned,
        disordered=disordered,
    )
    native["experimental"]["opxrd_path"] = rel
    cosine = native["experimental"].get("simulated_pxrd_cosine")
    status.update(
        {
            "status": "ok",
            "formula": native["formula"],
            "n_sites": native["n_sites"],
            "space_group": native["space_group"],
            "n_pxrd_bins": len(peak_x),
            "pattern_kind": "profile",
            "wavelength_A": wavelength,
            "q_aligned": q_aligned,
            "disordered_cif": disordered,
            "simulated_pxrd_cosine": cosine,
            "native": native,
        }
    )
    return status


def scored_mask(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    cosine = row.get("simulated_pxrd_cosine")
    if cosine is None or float(cosine) < COSINE_MIN:
        return False
    n_sites = int(row.get("n_sites") or 0)
    n_bins = int(row.get("n_pxrd_bins") or 0)
    return 2 <= n_sites <= MAX_SITES and n_bins >= MIN_PEAKS


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("formula") or ""), int(row.get("space_group") or 0))
        prev = best.get(key)
        cosine = float(row.get("simulated_pxrd_cosine") or -1.0)
        if prev is None or cosine > float(prev.get("simulated_pxrd_cosine") or -1.0):
            best[key] = row
    return [best[k] for k in sorted(best)]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_sequences(native_records: list[dict[str, Any]], path: Path) -> tuple[int, int]:
    from crystal_language.data.sequence import build_sequence_record

    seq_records = []
    failed = 0
    for record in native_records:
        if "atom_types" not in record and "atom_types" in record:
            record = dict(record)
            record["atom_types"] = record["atom_types"]
        try:
            seq_records.append(build_sequence_record(record, rep="v2"))
        except Exception:
            failed += 1
    write_jsonl(path, seq_records)
    return len(seq_records), failed


def plot_examples(records: list[dict[str, Any]], out_dir: Path, n: int = 8) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = records[:n]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.1 * len(chosen)), sharex=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, rec in zip(axes, chosen):
        spec = rec["conditions"]
        ax.vlines(spec["pxrd_x"], 0.0, spec["pxrd_y"], color="#d62728", linewidth=0.7, label="exp. PXRD")
        ax.vlines(spec["neutron_x"], 0.0, spec["neutron_y"], color="#1f77b4", linewidth=0.8, alpha=0.55, label="sim. ND")
        meta = rec["experimental"]
        cos = meta.get("simulated_pxrd_cosine")
        ax.set_ylabel("I")
        ax.set_xlim(GRID_MIN, GRID_MAX)
        ax.set_title(
            f"{rec['id']}  {rec['formula']}  n={rec['n_sites']}  "
            f"λ={meta.get('wavelength_A')}  cos={None if cos is None else round(cos, 3)}",
            loc="left",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("2θ (deg, training λ)")
    fig.tight_layout()
    fig.savefig(out_dir / "example_patterns.png", dpi=160)
    plt.close(fig)


def download_opxrd_zip(dest: Path) -> Path:
    """Prefer the labeled CNRS + HKUST-A slice over the 1.4 GB dump."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    labeled_dir = dest.parent / "opxrd"
    marker = labeled_dir / ".unpacked"
    if marker.exists() and any(labeled_dir.rglob("*.json")):
        return dest
    script = Path(__file__).with_name("download_opxrd_labeled.py")
    print(f"Downloading labeled opXRD subsets via {script}", flush=True)
    subprocess.check_call([sys.executable, str(script), "--out-dir", str(labeled_dir)])
    return dest


def unpack_opxrd(zip_path: Path, out_dir: Path) -> Path:
    marker = out_dir / ".unpacked"
    if marker.exists() and any(out_dir.rglob("*.json")):
        return out_dir
    if zip_path.exists() and zip_path.stat().st_size > 10_000_000:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Unpacking {zip_path} → {out_dir}", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        marker.write_text("ok\n", encoding="utf-8")
        return out_dir
    raise RuntimeError(
        f"opXRD JSON dir missing ({out_dir}). Run download_opxrd_labeled.py "
        "or pass --opxrd-dir."
    )


def iter_opxrd_json(root: Path) -> list[Path]:
    skip_names = {"download_summary.json", "extracted_index.json"}
    return sorted(
        p
        for p in root.rglob("*.json")
        if p.is_file()
        and p.stat().st_size > 0
        and p.name not in skip_names
        and p.parent.name != "opxrd_parts"
    )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "source",
        "status",
        "reason",
        "formula",
        "n_sites",
        "space_group",
        "n_pxrd_bins",
        "pattern_kind",
        "wavelength_A",
        "q_aligned",
        "disordered_cif",
        "simulated_pxrd_cosine",
        "year",
        "doi",
        "scored",
        "scored_unique",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-sites", type=int, default=MAX_SITES)
    parser.add_argument("--max-cod", type=int, default=None)
    parser.add_argument("--skip-cod", action="store_true")
    parser.add_argument("--opxrd-dir", default="")
    parser.add_argument("--download-opxrd", action="store_true")
    parser.add_argument("--max-opxrd", type=int, default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "cod"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    print("Wrote pre-registered protocol.json (before any parsing / eval).", flush=True)

    legal_elements = load_tokenizer_elements(TOKENIZER_PATH)
    results: list[dict[str, Any]] = []

    if not args.skip_cod:
        print("Querying COD X-ray powder + Fobs catalog …", flush=True)
        catalog = query_cod_catalog()
        if args.max_cod is not None:
            catalog = catalog[: args.max_cod]
        with (out_dir / "cod_catalog.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(catalog[0].keys()))
            writer.writeheader()
            writer.writerows(catalog)
        print(f"COD catalog: {len(catalog)} entries", flush=True)

        def _prefetch(row: dict[str, str]) -> None:
            cod_id = str(row["file"]).zfill(7)
            npd.download(f"{npd.COD_HTTP}/{cod_id}.cif", raw_dir / f"{cod_id}.cif")
            npd.download(f"{npd.COD_HTTP}/{cod_id}.hkl", raw_dir / f"{cod_id}.hkl")

        print("Downloading COD CIF/HKL …", flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(_prefetch, catalog))
        for i, row in enumerate(catalog, start=1):
            print(f"[COD {i}/{len(catalog)}] {row['file']}", flush=True)
            results.append(process_cod_entry(row, raw_dir, legal_elements, int(args.max_sites)))

    opxrd_dir = Path(args.opxrd_dir) if args.opxrd_dir else out_dir / "raw" / "opxrd"
    if args.download_opxrd:
        zip_path = download_opxrd_zip(out_dir / "raw" / "opxrd.zip")
        unpack_opxrd(zip_path, opxrd_dir)
    if opxrd_dir.exists():
        json_paths = iter_opxrd_json(opxrd_dir)
        if args.max_opxrd is not None:
            json_paths = json_paths[: args.max_opxrd]
        print(f"opXRD json files: {len(json_paths)}", flush=True)
        for i, path in enumerate(json_paths, start=1):
            if i % 200 == 0 or i == 1:
                print(f"[opXRD {i}/{len(json_paths)}] {path.name}", flush=True)
            results.append(process_opxrd_file(path, legal_elements, int(args.max_sites)))
    else:
        print(f"opXRD dir missing ({opxrd_dir}); COD-only this run.", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    scored = [r for r in ok if scored_mask(r)]
    unique = dedup_rows(scored)
    scored_ids = {r["id"] for r in scored}
    unique_ids = {r["id"] for r in unique}
    for row in results:
        row["scored"] = row.get("id") in scored_ids
        row["scored_unique"] = row.get("id") in unique_ids

    native_all = [r["native"] for r in ok]
    native_scored = [r["native"] for r in scored]
    native_unique = [r["native"] for r in unique]
    write_jsonl(out_dir / "native" / "test.jsonl", native_all)
    write_jsonl(out_dir / "native_scored" / "test.jsonl", native_scored)
    write_jsonl(out_dir / "native_scored_unique" / "test.jsonl", native_unique)
    n_seq, seq_failed = write_sequences(native_unique, out_dir / "crystal_language_v2" / "test.jsonl")
    write_sequences(native_all, out_dir / "crystal_language_v2_all" / "test.jsonl")
    plot_examples(native_unique, fig_dir)
    write_manifest(out_dir / "manifest.csv", results)

    skip_counts: Counter[str] = Counter()
    for row in results:
        if row.get("status") != "ok":
            reason = str(row.get("reason") or "unknown")
            skip_counts[reason.split(":")[0]] += 1

    summary = {
        "protocol": "public_exp_pxrd_v1",
        "n_attempted": len(results),
        "n_ok": len(ok),
        "n_scored": len(scored),
        "n_scored_unique": len(unique),
        "sequence_records": n_seq,
        "sequence_failed": seq_failed,
        "skip_reason_families": dict(skip_counts),
        "cosine_min": COSINE_MIN,
        "max_sites": int(args.max_sites),
        "sources": {
            "cod_ok": sum(1 for r in ok if r.get("source") == "cod"),
            "opxrd_ok": sum(1 for r in ok if r.get("source") == "opxrd"),
            "cod_scored_unique": sum(1 for r in unique if r.get("source") == "cod"),
            "opxrd_scored_unique": sum(1 for r in unique if r.get("source") == "opxrd"),
        },
        "notes": PROTOCOL["modalities"],
        "eval_example": (
            "python crystal_language/evaluate_generation.py "
            f"--data-dir {out_dir / 'crystal_language_v2'} "
            "--split test --condition-set pxrd_neutron_xrf --rerank "
            "--noise-profile none --rerank-match-sigma 0.40 --rerank-shift-max 0.20"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if unique else 1


if __name__ == "__main__":
    raise SystemExit(main())
