"""Parse canonical crystal language token sequences back into structure fields.

The renderer in ``data/sequence.py`` emits numbers as individual character
tokens. The parser regroups consecutive numeric-character tokens into floats and
reads the fixed block grammar. It returns a plain dict; conversion to a
``pymatgen.Structure`` is optional and only imported on demand.
"""

from __future__ import annotations

from typing import Any

from .tokenizer import angle_value, coord_value, length_value, space_group_value

NUM_CHARS = set("0123456789.-")


class ParseError(ValueError):
    pass


def _is_coord_token(token: str) -> bool:
    return token.startswith("<C") and token.endswith(">") and len(token) == 6


def _is_length_token(token: str) -> bool:
    return token.startswith("<L") and token.endswith(">") and len(token) == 7


def _is_angle_token(token: str) -> bool:
    return token.startswith("<G") and token.endswith(">") and len(token) == 6


def _is_sg_token(token: str) -> bool:
    return token.startswith("<SG") and token.endswith(">") and len(token) == 7


def _read_number(tokens: list[str], i: int) -> tuple[float, int]:
    chars: list[str] = []
    while i < len(tokens) and tokens[i] in NUM_CHARS and len(tokens[i]) == 1:
        chars.append(tokens[i])
        i += 1
    if not chars:
        raise ParseError(f"expected number at position {i}")
    text = "".join(chars)
    try:
        return float(text), i
    except ValueError as exc:
        raise ParseError(f"invalid number {text!r}") from exc


def _read_int(tokens: list[str], i: int) -> tuple[int, int]:
    value, i = _read_number(tokens, i)
    return int(round(value)), i


def _read_coord(tokens: list[str], i: int, axis: str) -> tuple[float, int]:
    """Read one coordinate: V2 bin token, or V1 ``axis`` marker + char number."""
    if i < len(tokens) and _is_coord_token(tokens[i]):
        try:
            return coord_value(tokens[i]), i + 1
        except ValueError as exc:
            raise ParseError(f"bad coord token {tokens[i]!r}") from exc
    i = _expect(tokens, i, axis)
    return _read_number(tokens, i)


def _expect(tokens: list[str], i: int, expected: str) -> int:
    if i >= len(tokens) or tokens[i] != expected:
        found = tokens[i] if i < len(tokens) else "<EOS-or-end>"
        raise ParseError(f"expected {expected!r} but found {found!r} at {i}")
    return i + 1


def _read_lattice(tokens: list[str], i: int) -> tuple[dict[str, float], int]:
    """Read a lattice block body (after '<LATTICE>') in V1 or V2 form."""
    lattice: dict[str, float] = {}
    if i < len(tokens) and _is_length_token(tokens[i]):
        keys = ("a", "b", "c", "alpha", "beta", "gamma")
        for k, key in enumerate(keys):
            if i >= len(tokens):
                raise ParseError(f"lattice block truncated at {key!r}")
            token = tokens[i]
            if k < 3:
                if not _is_length_token(token):
                    raise ParseError(f"expected length bin for {key!r}, got {token!r}")
                lattice[key] = length_value(token)
            else:
                if not _is_angle_token(token):
                    raise ParseError(f"expected angle bin for {key!r}, got {token!r}")
                lattice[key] = angle_value(token)
            i += 1
        return lattice, i
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        i = _expect(tokens, i, key)
        lattice[key], i = _read_number(tokens, i)
    return lattice, i


def parse_crystal_tokens(tokens: list[str]) -> dict[str, Any]:
    if "<CRYSTAL_START>" in tokens:
        start = tokens.index("<CRYSTAL_START>")
        tokens = tokens[start:]
    i = _expect(tokens, 0, "<CRYSTAL_START>")

    i = _expect(tokens, i, "<FORMULA>")
    composition: list[tuple[str, int]] = []
    while i < len(tokens) and tokens[i] not in {"<N_SITES>"}:
        element = tokens[i]
        i += 1
        count, i = _read_int(tokens, i)
        composition.append((element, count))

    i = _expect(tokens, i, "<N_SITES>")
    n_sites, i = _read_int(tokens, i)

    i = _expect(tokens, i, "<SPACE_GROUP>")
    if i < len(tokens) and _is_sg_token(tokens[i]):
        space_group = space_group_value(tokens[i])
        i += 1
    else:
        space_group, i = _read_int(tokens, i)

    i = _expect(tokens, i, "<LATTICE>")
    lattice, i = _read_lattice(tokens, i)

    i = _expect(tokens, i, "<ATOM_SITES_START>")
    sites: list[dict[str, Any]] = []
    while i < len(tokens) and tokens[i] == "<ATOM>":
        i += 1
        if i >= len(tokens):
            raise ParseError("expected element after '<ATOM>'")
        element = tokens[i]
        i += 1
        coord: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            coord[axis], i = _read_coord(tokens, i, axis)
        sites.append({"element": element, "frac": [coord["x"], coord["y"], coord["z"]]})
    i = _expect(tokens, i, "<ATOM_SITES_END>")
    i = _expect(tokens, i, "<CRYSTAL_END>")

    return {
        "composition": composition,
        "n_sites": n_sites,
        "space_group": space_group,
        "lattice": lattice,
        "sites": sites,
    }


def parse_site_tokens(tokens: list[str]) -> list[dict[str, Any]]:
    if "<ATOM_SITES_START>" in tokens:
        start = tokens.index("<ATOM_SITES_START>")
        tokens = tokens[start:]
    i = _expect(tokens, 0, "<ATOM_SITES_START>")
    sites: list[dict[str, Any]] = []
    while i < len(tokens) and tokens[i] == "<ATOM>":
        i += 1
        if i >= len(tokens):
            raise ParseError("expected element after '<ATOM>'")
        element = tokens[i]
        i += 1
        coord: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            coord[axis], i = _read_coord(tokens, i, axis)
        sites.append({"element": element, "frac": [coord["x"], coord["y"], coord["z"]]})
    _expect(tokens, i, "<ATOM_SITES_END>")
    return sites


def to_structure(parsed: dict[str, Any]):
    from pymatgen.core import Lattice, Structure

    latt = parsed["lattice"]
    lattice = Lattice.from_parameters(
        latt["a"], latt["b"], latt["c"], latt["alpha"], latt["beta"], latt["gamma"]
    )
    species = [site["element"] for site in parsed["sites"]]
    coords = [site["frac"] for site in parsed["sites"]]
    return Structure(lattice, species, coords)


def lattice_from_condition_tokens(tokens: list[str]) -> dict[str, float]:
    if "<LATTICE>" not in tokens:
        raise ParseError("known lattice block is missing '<LATTICE>'")
    i = tokens.index("<LATTICE>") + 1
    lattice, _ = _read_lattice(tokens, i)
    return lattice


def formula_from_condition_tokens(tokens: list[str]) -> list[tuple[str, int]]:
    if "<FORMULA>" not in tokens:
        raise ParseError("formula block is missing '<FORMULA>'")
    i = tokens.index("<FORMULA>") + 1
    composition: list[tuple[str, int]] = []
    while i < len(tokens) and tokens[i] not in {
        "<PXRD_START>",
        "<PXRD_END>",
        "<NEUTRON_START>",
        "<NEUTRON_END>",
        "<ELEMENT_SET>",
        "<XRF_START>",
        "<XRF_END>",
        "<XPS_VALENCE_START>",
        "<XPS_VALENCE_END>",
        "<COND_END>",
        "<KNOWN_LATTICE>",
        "<LATTICE>",
    }:
        element = tokens[i]
        i += 1
        count, i = _read_int(tokens, i)
        composition.append((element, count))
    return composition


def structure_from_sites(lattice: dict[str, float], sites: list[dict[str, Any]]):
    from pymatgen.core import Lattice, Structure

    pymatgen_lattice = Lattice.from_parameters(
        lattice["a"],
        lattice["b"],
        lattice["c"],
        lattice["alpha"],
        lattice["beta"],
        lattice["gamma"],
    )
    species = [site["element"] for site in sites]
    coords = [site["frac"] for site in sites]
    return Structure(pymatgen_lattice, species, coords)
