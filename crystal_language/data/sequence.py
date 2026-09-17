"""Canonical CIF-like crystal language sequence construction.

V1 ("char"): every numeric value is rendered as individual character tokens.
V2 ("bin"): coordinates / lattice / space group are single bin tokens
(`<C000>`-`<C999>`, `<L....>`, `<G...>`, `<SG...>`), spectra leave the token
stream entirely and are carried as raw peak arrays for a dense 1D encoder.
Both representations share the same deterministic parser-friendly grammar.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..tokenizer import (
    angle_token,
    coord_token,
    length_token,
    space_group_token,
)
from .peaks import extract_peaks

COORD_DECIMALS = 4
LATTICE_LEN_DECIMALS = 3
LATTICE_ANGLE_DECIMALS = 2
PEAK_POS_DECIMALS = 2
PEAK_INT_DECIMALS = 3
PEAK_WIDTH_DECIMALS = 3
XRF_DECIMALS = 3
MAX_PEAKS = 24
SPECTRA_ROUND_X = 3
SPECTRA_ROUND_Y = 3

CONDITION_SETS: dict[str, list[str]] = {
    "pxrd": ["pxrd"],
    "pxrd_element": ["pxrd", "element"],
    "pxrd_xrf": ["pxrd", "xrf"],
    "pxrd_neutron": ["pxrd", "neutron"],
    "pxrd_neutron_xrf": ["pxrd", "neutron", "xrf"],
    "pxrd_neutron_xrf_xps": ["pxrd", "neutron", "xrf", "xps"],
    "pxrd_formula": ["pxrd", "formula"],
}


def num_tokens(value: float, decimals: int) -> list[str]:
    text = f"{float(value):.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return list(text)


def int_tokens(value: int) -> list[str]:
    return list(str(int(value)))


def _composition(atom_types: list[str]) -> list[tuple[str, int]]:
    counts = Counter(atom_types)
    return sorted(counts.items(), key=lambda kv: kv[0])


def _formula_tokens(atom_types: list[str]) -> list[str]:
    tokens: list[str] = ["<FORMULA>"]
    for element, count in _composition(atom_types):
        tokens.append(element)
        tokens.extend(int_tokens(count))
    return tokens


def _sorted_sites(
    atom_types: list[str], frac_coords: list[list[float]]
) -> list[tuple[str, float, float, float]]:
    sites: list[tuple[str, float, float, float]] = []
    for element, coord in zip(atom_types, frac_coords):
        x = round(float(coord[0]) % 1.0, COORD_DECIMALS)
        y = round(float(coord[1]) % 1.0, COORD_DECIMALS)
        z = round(float(coord[2]) % 1.0, COORD_DECIMALS)
        sites.append((element, x, y, z))
    sites.sort(key=lambda s: (s[0], s[1], s[2], s[3]))
    return sites


def target_tokens(record: dict[str, Any], rep: str = "v1") -> list[str]:
    atom_types = [str(a) for a in record["atom_types"]]
    frac_coords = record["frac_coords"]
    params = record.get("lattice_parameters") or {}
    lengths = params.get("lengths") or [0.0, 0.0, 0.0]
    angles = params.get("angles") or [90.0, 90.0, 90.0]
    space_group = int(record.get("space_group") or 0)

    tokens: list[str] = ["<CRYSTAL_START>"]
    tokens.extend(_formula_tokens(atom_types))
    tokens.append("<N_SITES>")
    tokens.extend(int_tokens(len(atom_types)))
    tokens.append("<SPACE_GROUP>")
    if rep == "v2":
        tokens.append(space_group_token(max(1, space_group)))
    else:
        tokens.extend(int_tokens(space_group))
    tokens.append("<LATTICE>")
    if rep == "v2":
        tokens.extend(length_token(v) for v in lengths[:3])
        tokens.extend(angle_token(v) for v in angles[:3])
    else:
        tokens.append("a")
        tokens.extend(num_tokens(lengths[0], LATTICE_LEN_DECIMALS))
        tokens.append("b")
        tokens.extend(num_tokens(lengths[1], LATTICE_LEN_DECIMALS))
        tokens.append("c")
        tokens.extend(num_tokens(lengths[2], LATTICE_LEN_DECIMALS))
        tokens.append("alpha")
        tokens.extend(num_tokens(angles[0], LATTICE_ANGLE_DECIMALS))
        tokens.append("beta")
        tokens.extend(num_tokens(angles[1], LATTICE_ANGLE_DECIMALS))
        tokens.append("gamma")
        tokens.extend(num_tokens(angles[2], LATTICE_ANGLE_DECIMALS))
    tokens.extend(site_tokens(record, rep=rep))
    tokens.append("<CRYSTAL_END>")
    return tokens


def site_tokens(record: dict[str, Any], rep: str = "v1") -> list[str]:
    atom_types = [str(a) for a in record["atom_types"]]
    frac_coords = record["frac_coords"]

    tokens: list[str] = ["<ATOM_SITES_START>"]
    for element, x, y, z in _sorted_sites(atom_types, frac_coords):
        tokens.append("<ATOM>")
        tokens.append(element)
        if rep == "v2":
            tokens.append(coord_token(x))
            tokens.append(coord_token(y))
            tokens.append(coord_token(z))
        else:
            tokens.append("x")
            tokens.extend(num_tokens(x, COORD_DECIMALS))
            tokens.append("y")
            tokens.extend(num_tokens(y, COORD_DECIMALS))
            tokens.append("z")
            tokens.extend(num_tokens(z, COORD_DECIMALS))
    tokens.append("<ATOM_SITES_END>")
    return tokens


def _peak_block(start: str, end: str, x, y) -> list[str]:
    peaks = extract_peaks(x, y, max_peaks=MAX_PEAKS)
    if not peaks:
        return []
    tokens: list[str] = [start]
    for peak in peaks:
        tokens.append("<PEAK>")
        tokens.append("t")
        tokens.extend(num_tokens(peak.two_theta, PEAK_POS_DECIMALS))
        tokens.append("i")
        tokens.extend(num_tokens(peak.intensity, PEAK_INT_DECIMALS))
        tokens.append("w")
        tokens.extend(num_tokens(peak.width, PEAK_WIDTH_DECIMALS))
    tokens.append(end)
    return tokens


def _element_block(atom_types: list[str]) -> list[str]:
    elements = sorted({str(a) for a in atom_types})
    if not elements:
        return []
    return ["<ELEMENT_SET>", *elements]


def _xrf_block(xrf: dict[str, float] | None) -> list[str]:
    if not xrf:
        return []
    tokens: list[str] = ["<XRF_START>"]
    for element, frac in sorted(xrf.items(), key=lambda kv: kv[0]):
        tokens.append("<XRF_MASS_FRAC>")
        tokens.append(element)
        tokens.extend(num_tokens(frac, XRF_DECIMALS))
    tokens.append("<XRF_END>")
    return tokens


def _xps_block(valence: dict[str, float] | None) -> list[str]:
    if not valence:
        return []
    tokens: list[str] = ["<XPS_VALENCE_START>"]
    for element, state in sorted(valence.items(), key=lambda kv: kv[0]):
        tokens.append("<VALENCE>")
        tokens.append(element)
        tokens.extend(num_tokens(state, 1))
    tokens.append("<XPS_VALENCE_END>")
    return tokens


def _lattice_block(record: dict[str, Any], rep: str = "v1") -> list[str]:
    params = record.get("lattice_parameters") or {}
    lengths = params.get("lengths") or [0.0, 0.0, 0.0]
    angles = params.get("angles") or [90.0, 90.0, 90.0]
    tokens: list[str] = ["<KNOWN_LATTICE>", "<LATTICE>"]
    if rep == "v2":
        tokens.extend(length_token(v) for v in lengths[:3])
        tokens.extend(angle_token(v) for v in angles[:3])
        return tokens
    tokens.append("a")
    tokens.extend(num_tokens(lengths[0], LATTICE_LEN_DECIMALS))
    tokens.append("b")
    tokens.extend(num_tokens(lengths[1], LATTICE_LEN_DECIMALS))
    tokens.append("c")
    tokens.extend(num_tokens(lengths[2], LATTICE_LEN_DECIMALS))
    tokens.append("alpha")
    tokens.extend(num_tokens(angles[0], LATTICE_ANGLE_DECIMALS))
    tokens.append("beta")
    tokens.extend(num_tokens(angles[1], LATTICE_ANGLE_DECIMALS))
    tokens.append("gamma")
    tokens.extend(num_tokens(angles[2], LATTICE_ANGLE_DECIMALS))
    return tokens


def condition_blocks(record: dict[str, Any], rep: str = "v1") -> dict[str, list[str]]:
    atom_types = [str(a) for a in record["atom_types"]]
    conditions = record.get("conditions") or {}
    blocks: dict[str, list[str]] = {}

    if rep != "v2":
        pxrd = _peak_block(
            "<PXRD_START>", "<PXRD_END>",
            conditions.get("pxrd_x"), conditions.get("pxrd_y"),
        )
        if pxrd:
            blocks["pxrd"] = pxrd

        neutron = _peak_block(
            "<NEUTRON_START>", "<NEUTRON_END>",
            conditions.get("neutron_x"), conditions.get("neutron_y"),
        )
        if neutron:
            blocks["neutron"] = neutron

    element = _element_block(atom_types)
    if element:
        blocks["element"] = element

    xrf = _xrf_block(conditions.get("xrf_mass_frac"))
    if xrf:
        blocks["xrf"] = xrf

    xps = _xps_block(conditions.get("valence_assignments"))
    if xps:
        blocks["xps"] = xps

    blocks["formula"] = _formula_tokens(atom_types)
    blocks["lattice"] = _lattice_block(record, rep=rep)
    return blocks


def _spectra_arrays(record: dict[str, Any]) -> dict[str, list[float]]:
    """Full discrete peak lists for the dense spectra encoder (V2 only)."""
    conditions = record.get("conditions") or {}
    out: dict[str, list[float]] = {}
    for name in ("pxrd", "neutron"):
        x = conditions.get(f"{name}_x")
        y = conditions.get(f"{name}_y")
        if not x or not y or len(x) != len(y):
            continue
        ymax = max(float(v) for v in y)
        if ymax <= 0:
            continue
        out[f"{name}_x"] = [round(float(v), SPECTRA_ROUND_X) for v in x]
        out[f"{name}_y"] = [round(float(v) / ymax, SPECTRA_ROUND_Y) for v in y]
    return out


def build_sequence_record(record: dict[str, Any], rep: str = "v1") -> dict[str, Any]:
    blocks = condition_blocks(record, rep=rep)
    target = target_tokens(record, rep=rep)
    sites = site_tokens(record, rep=rep)
    out = {
        "id": record.get("id"),
        "source": record.get("source"),
        "split": record.get("split"),
        "n_sites": int(record.get("n_sites") or len(record["atom_types"])),
        "target": " ".join(target),
        "sites_target": " ".join(sites),
        "cond": {name: " ".join(tokens) for name, tokens in blocks.items()},
    }
    if rep == "v2":
        out["rep"] = "v2"
        out["spectra"] = _spectra_arrays(record)
    return out
