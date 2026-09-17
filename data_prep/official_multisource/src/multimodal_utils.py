"""Multimodal helper utilities used during dataset construction.

This module simulates additional characterisation signals from a crystal
structure to support the AI-empowered X-ray / neutron / XRF / XPS framework:

- ``compute_neutron_pattern``    : powder neutron diffraction (Cu-K-alpha-like
  scan range used by default to stay aligned with PXRD).
- ``compute_neutron_pdf``        : optional pair distribution function
  (computed only when explicitly requested because it is several times slower).
- ``compute_valence``            : XPS-style oxidation-state assignment based on
  pymatgen's compositional guesser with deterministic fallbacks.

All functions are best-effort: on failure they return ``None`` (or empty
mappings) so the caller can still emit a record with the field marked as
missing.  The training-time data loader is responsible for converting these
``None`` markers into ``present_flag = 0`` features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.core import Composition, Structure


def compute_neutron_pattern(
    structure: Structure,
    two_theta_min: float = 5.0,
    two_theta_max: float = 80.0,
    wavelength: float = 1.54184,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Simulate a powder neutron diffraction pattern.

    Returns (two_theta, intensity) or ``None`` if the calculator fails (e.g. the
    structure contains an isotope without a tabulated coherent scattering
    length).
    """
    try:
        calc = NDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(
            structure,
            scaled=True,
            two_theta_range=(two_theta_min, two_theta_max),
        )
    except Exception:
        return None
    x = np.asarray(pattern.x, dtype=np.float64)
    y = np.asarray(pattern.y, dtype=np.float64)
    if x.size == 0:
        return None
    return x, y


def compute_valence(structure: Structure, max_sites_for_guess: int = 24) -> dict[str, Any]:
    """Return XPS-like oxidation-state metadata for a structure.

    The output dict has three keys:

    - ``valence_assignments``: ``{element_symbol: mean_oxidation_state}``
    - ``valence_source``:      ``"oxi_state_guesses"``, ``"common_oxidation_states"``
                               or ``"unknown"``
    - ``valence_assigned``:    ``True`` if at least one element got a value.

    We avoid running the heavy SMACT/oxi-guess procedure on huge unit cells
    (rare in MP/WBM but possible in AFLOW protocols) by checking ``max_sites_for_guess``
    before attempting :py:meth:`Composition.oxi_state_guesses`.  For larger
    structures we fall back to elemental common oxidation states.
    """
    composition = structure.composition
    assignments: dict[str, float] = {}
    source = "unknown"
    if len(structure) <= max_sites_for_guess:
        try:
            guesses = composition.oxi_state_guesses()
        except Exception:
            guesses = ()
        if guesses:
            for el_symbol, ox in guesses[0].items():
                assignments[str(el_symbol)] = float(ox)
            source = "oxi_state_guesses"
    if not assignments:
        for el in composition.elements:
            cos = list(getattr(el, "common_oxidation_states", ()) or [])
            if not cos:
                cos = list(getattr(el, "oxidation_states", ()) or [])
            if cos:
                # Most common state is conventionally listed first.
                assignments[el.symbol] = float(cos[0])
        if assignments:
            source = "common_oxidation_states"
    return {
        "valence_assignments": assignments,
        "valence_source": source,
        "valence_assigned": bool(assignments),
    }


def compute_multimodal_signals(
    structure: Structure,
    two_theta_min: float = 5.0,
    two_theta_max: float = 80.0,
    enable_neutron: bool = True,
    enable_valence: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper used by ``structure_to_training_record``.

    Returns a record-fragment dict that is safe to merge into the main record;
    missing fields are left as ``None`` / empty so the model side can detect
    them via a present-flag.
    """
    out: dict[str, Any] = {
        "neutron_x": None,
        "neutron_y": None,
        "valence_assignments": None,
        "valence_source": None,
        "valence_assigned": False,
    }
    if enable_neutron:
        nd = compute_neutron_pattern(structure, two_theta_min, two_theta_max)
        if nd is not None:
            out["neutron_x"], out["neutron_y"] = nd
    if enable_valence:
        out.update(compute_valence(structure))
    return out
