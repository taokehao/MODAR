from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .multimodal_utils import compute_multimodal_signals


def as_primitive_structure(structure: Structure) -> Structure:
    """Return a reduced primitive structure suitable for Uni-3DAR training."""
    try:
        structure = SpacegroupAnalyzer(structure, symprec=0.1).get_primitive_standard_structure()
    except Exception:
        structure = structure.get_primitive_structure()
    return structure.get_reduced_structure()


def compute_pxrd(
    structure: Structure,
    two_theta_min: float = 5.0,
    two_theta_max: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    pattern = XRDCalculator().get_pattern(
        structure,
        scaled=True,
        two_theta_range=(two_theta_min, two_theta_max),
    )
    return np.asarray(pattern.x, dtype=np.float64), np.asarray(pattern.y, dtype=np.float64)


def structure_to_training_record(
    *,
    structure: Structure,
    source: str,
    source_id: str,
    formation_energy_per_atom: float | None,
    e_above_hull: float | None,
    band_gap: float | None,
    two_theta_min: float = 5.0,
    two_theta_max: float = 80.0,
    enable_neutron: bool = True,
    enable_valence: bool = True,
    multimodal: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primitive = as_primitive_structure(structure)
    pxrd_x, pxrd_y = compute_pxrd(primitive, two_theta_min, two_theta_max)
    record: dict[str, Any] = {
        # Core fields consumed by the model dataloader.
        "atom_type": [site.specie.symbol for site in primitive],
        "atom_pos": np.asarray(primitive.frac_coords, dtype=np.float64),
        "lattice_matrix": np.asarray(primitive.lattice.matrix, dtype=np.float64),
        "pxrd_x": pxrd_x,
        "pxrd_y": pxrd_y,
        # Provenance + auxiliary property fields.
        "source": source,
        "source_id": str(source_id),
        "formula": primitive.composition.reduced_formula,
        "pretty_formula": primitive.composition.reduced_formula,
        "n_sites": int(len(primitive)),
        "formation_energy_per_atom": _finite_or_none(formation_energy_per_atom),
        "e_above_hull": _hull_or_none(e_above_hull),
        "band_gap": _finite_or_none(band_gap),
        "bvse_li_e1d": None,
        "bvse_na_e1d": None,
    }
    # Multimodal training signals (neutron diffraction + XPS-style valences).
    # If an offline enrichment step already attached these fields to the raw
    # JSONL row, reuse them so repeated LMDB builds are cheap.
    if multimodal is None:
        mm = compute_multimodal_signals(
            primitive,
            two_theta_min=two_theta_min,
            two_theta_max=two_theta_max,
            enable_neutron=enable_neutron,
            enable_valence=enable_valence,
        )
    else:
        mm = {
            "neutron_x": _array_or_none(multimodal.get("neutron_x")),
            "neutron_y": _array_or_none(multimodal.get("neutron_y")),
            "valence_assignments": multimodal.get("valence_assignments"),
            "valence_source": multimodal.get("valence_source"),
            "valence_assigned": bool(multimodal.get("valence_assigned")),
        }
    record.update(mm)
    try:
        record["space_group"] = int(SpacegroupAnalyzer(primitive, symprec=0.1).get_space_group_number())
    except Exception:
        record["space_group"] = None
    record["structure_hash"] = structure_hash(primitive)
    if extra:
        record.update(extra)
    return record


def structure_hash(structure: Structure) -> str:
    """Stable-enough dedup key based on composition, lengths/angles and fractional coords."""
    s = as_primitive_structure(structure)
    species = [site.specie.symbol for site in s]
    coords = np.mod(np.asarray(s.frac_coords, dtype=float), 1.0)
    order = sorted(range(len(species)), key=lambda i: (species[i], *np.round(coords[i], 4)))
    payload = {
        "formula": s.composition.reduced_formula,
        "species": [species[i] for i in order],
        "coords": np.round(coords[order], 4).tolist(),
        "lengths": np.round(s.lattice.abc, 3).tolist(),
        "angles": np.round(s.lattice.angles, 2).tolist(),
    }
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _hull_or_none(value: float | None) -> float | None:
    value = _finite_or_none(value)
    if value is None:
        return None
    # Some public summary tables contain small/medium negative hull distances
    # after correction. For conditioning, use physical distance to hull.
    return max(0.0, value)


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr
