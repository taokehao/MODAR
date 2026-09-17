#!/usr/bin/env python
"""Build a first-round experimental neutron-powder eval set from COD.

COD has ~900 neutron *powder structures*, but only a few dozen deposit an
observed diffractogram (``has Fobs``). This script:

1. queries those entries from the public COD MySQL
2. downloads CIF + optional HKL over HTTP
3. parses pdCIF profiles or Bragg peak lists
4. maps experimental 2θ/d/Q into Q-space, then onto the training 2θ grid
   (λ=1.54184 Å, 5–80°)
5. rasterizes background-subtracted profiles (or peak-picks sparse HKL lists)
6. simulates PXRD / XRF / XPS from the CIF structure
7. writes crystal-native JSONL + V2 sequence JSONL for ``evaluate_generation.py``

Deuterium is mapped to H in the language-model target (the tokenizer has no
``D`` token). Experimental neutron intensities are Q-aligned onto the training
wavelength; they are not left on the native 2θ axis.

Usage (LLMD env)::

  python data_prep/official_multisource/scripts/build_cod_experimental_npd.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

COD_HTTP = "http://www.crystallography.net/cod"
COD_MYSQL = [
    "mysql",
    "-h",
    "sql.crystallography.net",
    "-u",
    "cod_reader",
    "--batch",
    "--raw",
    "--connect-timeout=20",
]
GRID_MIN = 5.0
GRID_MAX = 80.0
GRID_STEP = 0.05
REF_WAVELENGTH = 1.54184  # project default (Cu-Kα-like)
MAX_SITES = 80  # experimental-noise probe; 120-site cells are out of scope
MIN_PEAKS = 8
PEAK_REL_THRESHOLD = 0.02
D_TO_H = True

DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "experimental_npd_cod"

TWOTHETA_TAGS = (
    "_pd_proc_2theta_corrected",
    "_pd_meas_2theta_scan",
    "_pd_proc_2theta_scan",
    "_pd_meas_2theta_fixed",
)
D_TAGS = ("_pd_proc_d_spacing", "_pd_meas_d_spacing", "_refln_d_spacing")
Q_TAGS = ("_pd_proc_recip_len_Q", "_pd_meas_recip_len_Q")
INTENSITY_TAGS = (
    "_pd_proc_intensity_net",
    "_pd_meas_intensity_net",
    "_pd_proc_intensity_total",
    "_pd_meas_intensity_total",
    "_pd_meas_counts_total",
    "_pd_meas_counts",
    "_pd_proc_intensity_corr",
    "_gsas_i100_meas",
    "_pd_calc_intensity_net",
    "_pd_calc_intensity_total",
)
BKG_TAGS = ("_pd_proc_intensity_bkg_calc", "_pd_meas_intensity_background")
F2_TAGS = ("_refln_f_squared_meas", "_refln_f_meas", "_refln_intensity_meas")
MULT_TAGS = ("_refln_mult",)
WAVELENGTH_TAGS = (
    "_diffrn_radiation_wavelength",
    "_pd_proc_wavelength",
    "_pd_instr_wavelength",
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {".", "?", "None"}:
            return None
        text = re.sub(r"\([^)]*\)", "", text)
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def _parse_wavelength(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str) and re.search(r"<|lambda", raw, flags=re.I) and _finite(raw) is None:
        # TOF / bandpass strings such as '0.36 < lambda < 10.18 Ang'.
        return None
    value = _finite(raw)
    if value is not None and 0.4 < value < 4.5:
        return value
    if not isinstance(raw, str):
        return None
    nums = [_finite(x) for x in re.findall(r"[0-9]+\.[0-9]+", raw)]
    nums = [x for x in nums if x is not None and 0.4 < x < 4.5]
    if len(nums) != 1:
        return None
    return float(nums[0])


def two_theta_from_d(d: float, wavelength: float) -> float | None:
    arg = wavelength / (2.0 * d)
    if not (0.0 < arg < 1.0):
        return None
    return math.degrees(2.0 * math.asin(arg))


def two_theta_from_q(q: float, wavelength: float) -> float | None:
    # Q = 4π sinθ / λ  →  2θ = 2 arcsin(Q λ / 4π)
    arg = q * wavelength / (4.0 * math.pi)
    if not (0.0 < arg < 1.0):
        return None
    return math.degrees(2.0 * math.asin(arg))


def q_from_two_theta(two_theta_deg: float, wavelength: float) -> float | None:
    if wavelength <= 0:
        return None
    s = math.sin(math.radians(two_theta_deg) / 2.0)
    if s <= 0:
        return None
    return 4.0 * math.pi * s / wavelength


def remap_two_theta_to_train(
    xs: np.ndarray,
    ys: np.ndarray,
    lambda_native: float,
    lambda_train: float = REF_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Map experimental 2θ at λ_native onto 2θ at the training wavelength.

    Same Q, different λ:  Q = 4π sinθ / λ.  Without this, a Cu-Kα-trained
    encoder reads CW patterns at λ≠1.54184 Å as if the 2θ numbers were already
    on the training axis.
    """
    if (
        lambda_native is None
        or lambda_native <= 0
        or abs(float(lambda_native) - lambda_train) < 0.02
    ):
        return xs, ys, False
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        q = q_from_two_theta(float(x), float(lambda_native))
        if q is None:
            continue
        tt = two_theta_from_q(q, lambda_train)
        if tt is None:
            continue
        out_x.append(float(tt))
        out_y.append(float(y))
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float), True


def _rolling_percentile(y: np.ndarray, window: int, q: float = 15.0) -> np.ndarray:
    if y.size == 0:
        return y
    window = max(5, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.percentile(views, q, axis=1)


def _tokenize_cif(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "#":
            while i < n and text[i] not in "\n\r":
                i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if c == ";" and (i == 0 or text[i - 1] in "\n\r"):
            i += 1
            start = i
            while i < n:
                if text[i] == ";" and text[i - 1] in "\n\r":
                    break
                i += 1
            tokens.append(text[start:i].strip())
            if i < n and text[i] == ";":
                i += 1
            continue
        if c in "\"'":
            quote = c
            i += 1
            start = i
            while i < n and text[i] != quote:
                i += 1
            tokens.append(text[start:i])
            i += 1
            continue
        start = i
        while i < n and not text[i].isspace():
            i += 1
        tokens.append(text[start:i])
    return tokens


def parse_cif_blocks(text: str) -> list[dict[str, Any]]:
    tokens = _tokenize_cif(text)
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.lower().startswith("data_"):
            current = {"name": tok[5:] or f"block{len(blocks)}", "items": {}, "loops": []}
            blocks.append(current)
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if tok.lower() == "loop_":
            i += 1
            tags: list[str] = []
            while i < len(tokens) and tokens[i].startswith("_"):
                tags.append(tokens[i].lower())
                i += 1
            values: list[str] = []
            while i < len(tokens):
                nxt = tokens[i]
                if nxt.lower() in {"loop_", "global_"} or nxt.lower().startswith("data_"):
                    break
                if nxt.startswith("_") and (
                    i + 1 >= len(tokens) or not tokens[i + 1].startswith("_")
                ):
                    # Possible start of a key/value pair. If the next token is
                    # not another tag, treat this as the end of the loop.
                    if i + 1 < len(tokens) and not tokens[i + 1].lower().startswith(
                        ("loop_", "data_", "global_")
                    ):
                        break
                values.append(nxt)
                i += 1
            n_tags = len(tags)
            loop: dict[str, list[str]] = {tag: [] for tag in tags}
            if n_tags:
                usable = values
                drop_point_id = False
                if tags and tags[0].endswith("point_id") and n_tags >= 3 and values:
                    first = _finite(values[0])
                    if first is not None and first < 400 and len(values) % (n_tags - 1) == 0:
                        drop_point_id = True
                if drop_point_id:
                    tags = tags[1:]
                    n_tags = len(tags)
                    loop = {tag: [] for tag in tags}
                if n_tags and len(usable) >= n_tags:
                    n_rows = len(usable) // n_tags
                    usable = usable[: n_rows * n_tags]
                    for r in range(n_rows):
                        for c, tag in enumerate(tags):
                            loop[tag].append(usable[r * n_tags + c])
            if any(loop.values()):
                current["loops"].append(loop)
            continue
        if tok.startswith("_"):
            key = tok.lower()
            i += 1
            val = tokens[i] if i < len(tokens) else ""
            if i < len(tokens):
                i += 1
            current["items"][key] = val
            continue
        i += 1
    return blocks


def _loop_column(loop: dict[str, list[str]], tags: tuple[str, ...]) -> list[float | None] | None:
    for tag in tags:
        if tag in loop:
            return [_finite(v) for v in loop[tag]]
    return None


def _block_wavelength(block: dict[str, Any], fallback: float | None = None) -> float | None:
    items = block.get("items") or {}
    for tag in WAVELENGTH_TAGS:
        parsed = _parse_wavelength(items.get(tag))
        if parsed is not None:
            return parsed
    for loop in block.get("loops") or []:
        for tag in WAVELENGTH_TAGS:
            if tag in loop:
                for raw in loop[tag]:
                    parsed = _parse_wavelength(raw)
                    if parsed is not None:
                        return parsed
    return fallback


def _is_neutron_block(block: dict[str, Any]) -> bool:
    blob = " ".join(str(v) for v in (block.get("items") or {}).values()).lower()
    return "neutron" in blob


def _reconstruct_two_theta(items: dict[str, str], n_points: int) -> list[float] | None:
    vmin = _finite(items.get("_pd_proc_2theta_range_min") or items.get("_pd_meas_2theta_range_min"))
    vinc = _finite(items.get("_pd_proc_2theta_range_inc") or items.get("_pd_meas_2theta_range_inc"))
    if vmin is None or vinc is None or vinc <= 0 or n_points <= 0:
        return None
    return [vmin + i * vinc for i in range(n_points)]


def _pattern_from_loop(
    loop: dict[str, list[str]],
    wavelength: float,
    items: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    intensity = _loop_column(loop, INTENSITY_TAGS)
    if intensity is None:
        f2 = _loop_column(loop, F2_TAGS)
        if f2 is None:
            return None
        mult = _loop_column(loop, MULT_TAGS) or [1.0] * len(f2)
        intensity = [
            (None if a is None else abs(a) * (1.0 if b is None else abs(b)))
            for a, b in zip(f2, mult)
        ]
        kind = "hkl"
    else:
        bkg = _loop_column(loop, BKG_TAGS)
        if bkg is not None and len(bkg) == len(intensity):
            intensity = [
                None
                if obs is None
                else max(0.0, obs - (0.0 if bg is None else bg))
                for obs, bg in zip(intensity, bkg)
            ]
        kind = "profile"

    two_theta = _loop_column(loop, TWOTHETA_TAGS)
    if two_theta is None:
        d_vals = _loop_column(loop, D_TAGS)
        if d_vals is not None:
            two_theta = [
                None if d is None else two_theta_from_d(d, wavelength) for d in d_vals
            ]
            kind = kind + "+d"
        else:
            q_vals = _loop_column(loop, Q_TAGS)
            if q_vals is not None:
                two_theta = [
                    None if q is None else two_theta_from_q(q, wavelength) for q in q_vals
                ]
                kind = kind + "+q"
            else:
                reconstructed = _reconstruct_two_theta(items or {}, len(intensity))
                if reconstructed is None:
                    return None
                two_theta = reconstructed
                kind = kind + "+range"

    xs: list[float] = []
    ys: list[float] = []
    for x, y in zip(two_theta, intensity):
        if x is None or y is None:
            continue
        if y <= 0:
            continue
        if not (0.05 < x < 179.0):
            continue
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < MIN_PEAKS:
        return None
    return np.asarray(xs), np.asarray(ys), kind


def extract_experimental_pattern(
    texts: list[str],
    catalog_wavelength: float | None,
) -> dict[str, Any] | None:
    blocks: list[dict[str, Any]] = []
    for text in texts:
        blocks.extend(parse_cif_blocks(text))
    if not blocks:
        return None

    wavelength = catalog_wavelength
    if wavelength is None:
        for block in blocks:
            wavelength = _block_wavelength(block, wavelength)
    if wavelength is None:
        wavelength = REF_WAVELENGTH

    merged_items: dict[str, str] = {}
    for block in blocks:
        merged_items.update(block.get("items") or {})

    ranked = sorted(
        blocks,
        key=lambda b: (
            int(_is_neutron_block(b)),
            max((len(next(iter(loop.values()))) for loop in b.get("loops") or [] if loop), default=0),
        ),
        reverse=True,
    )
    best: dict[str, Any] | None = None
    for block in ranked:
        items = {**merged_items, **(block.get("items") or {})}
        for loop in block.get("loops") or []:
            parsed = _pattern_from_loop(loop, wavelength, items)
            if parsed is None:
                continue
            xs, ys, kind = parsed
            score = int("profile" in kind) * 1_000_000 + int(_is_neutron_block(block)) * 1000 + len(xs)
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


def clip_and_peaks(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_min: float = GRID_MIN,
    grid_max: float = GRID_MAX,
    *,
    dense_profile: bool = False,
) -> tuple[list[float], list[float], list[float], list[float]]:
    from crystal_language.data.peaks import extract_peaks

    mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    x_clip = xs[mask]
    y_clip = ys[mask]
    mask = (x_clip >= grid_min) & (x_clip <= grid_max)
    x_clip = x_clip[mask]
    y_clip = y_clip[mask]
    if x_clip.size == 0:
        return [], [], [], []
    order = np.argsort(x_clip)
    x_clip = x_clip[order]
    y_clip = y_clip[order]
    y_clip = y_clip / float(np.max(y_clip))

    use_profile = dense_profile or x_clip.size > 200
    if use_profile:
        n = int(round((grid_max - grid_min) / GRID_STEP)) + 1
        acc = np.zeros(n, dtype=float)
        idx = np.round((x_clip - grid_min) / GRID_STEP).astype(int)
        keep_idx = (idx >= 0) & (idx < n)
        for i, y in zip(idx[keep_idx], y_clip[keep_idx]):
            if y > acc[i]:
                acc[i] = float(y)
        bkg = _rolling_percentile(acc, window=max(21, n // 25), q=15.0)
        net = np.clip(acc - bkg, 0.0, None)
        peak = float(net.max())
        if peak <= 0:
            net = acc
            peak = float(net.max())
        if peak <= 0:
            return x_clip.tolist(), y_clip.tolist(), [], []
        net = net / peak
        grid = grid_min + np.arange(n) * GRID_STEP
        keep = net >= 0.02
        if int(keep.sum()) < MIN_PEAKS:
            keep = net >= 0.005
        return x_clip.tolist(), y_clip.tolist(), grid[keep].tolist(), net[keep].tolist()

    rel = max(PEAK_REL_THRESHOLD, 0.04)
    peaks = extract_peaks(
        x_clip.tolist(),
        y_clip.tolist(),
        max_peaks=80,
        rel_threshold=rel,
    )
    peak_x = [p.two_theta for p in peaks]
    peak_y = [p.intensity for p in peaks]
    kept: list[tuple[float, float]] = []
    for x, y in sorted(zip(peak_x, peak_y), key=lambda t: t[1], reverse=True):
        if all(abs(x - kx) >= 0.25 for kx, _ in kept):
            kept.append((x, y))
    kept.sort(key=lambda t: t[0])
    peak_x = [a for a, _ in kept]
    peak_y = [b for _, b in kept]
    return x_clip.tolist(), y_clip.tolist(), peak_x, peak_y


def query_cod_catalog() -> list[dict[str, str]]:
    sql = """
SELECT file, formula, calcformula, nel, sg, sgNumber, wavelength, radiation,
       radType, flags, journal, year, doi, title
FROM data
WHERE method='powder diffraction'
  AND (radiation LIKE '%neutron%' OR radType LIKE '%neutron%')
  AND flags LIKE '%has Fobs%'
ORDER BY year DESC, file;
"""
    proc = subprocess.run(
        [*COD_MYSQL, "-e", sql, "cod"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("COD query returned no rows")
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(dict(zip(header, parts)))
    return rows


def download(url: str, dest: Path, retries: int = 4, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "multimodal_csp-v2/experimental-npd"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("empty body")
            dest.write_bytes(data)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_err = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                return False
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_err = exc
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    if last_err is not None:
        print(f"download failed {url}: {last_err}", file=sys.stderr)
    return False


def site_symbol(site: Any) -> str:
    if getattr(site, "is_ordered", False):
        symbol = site.specie.symbol
    else:
        species = getattr(site, "species", None)
        if species is None:
            symbol = str(site.specie)
        else:
            symbol = max(species.items(), key=lambda kv: kv[1])[0].symbol
    if D_TO_H and symbol == "D":
        return "H"
    return symbol


def _normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip()
    if D_TO_H and symbol == "D":
        return "H"
    return symbol


def structure_from_loops(text: str) -> Any:
    from pymatgen.core import Lattice, Structure

    blocks = parse_cif_blocks(text)
    if not blocks:
        raise ValueError("no CIF blocks")
    items: dict[str, str] = {}
    atom_loop = None
    for block in blocks:
        items.update(block.get("items") or {})
        for loop in block.get("loops") or []:
            if "_atom_site_fract_x" in loop and (
                "_atom_site_type_symbol" in loop or "_atom_site_label" in loop
            ):
                atom_loop = loop
    if atom_loop is None:
        raise ValueError("no atom_site loop")
    a = _finite(items.get("_cell_length_a"))
    b = _finite(items.get("_cell_length_b")) or a
    c = _finite(items.get("_cell_length_c")) or a
    alpha = _finite(items.get("_cell_angle_alpha")) or 90.0
    beta = _finite(items.get("_cell_angle_beta")) or 90.0
    gamma = _finite(items.get("_cell_angle_gamma")) or 90.0
    if a is None:
        raise ValueError("missing cell lengths")
    n = len(next(iter(atom_loop.values())))
    symbols = atom_loop.get("_atom_site_type_symbol") or atom_loop.get("_atom_site_label") or []
    xs = atom_loop.get("_atom_site_fract_x") or []
    ys = atom_loop.get("_atom_site_fract_y") or []
    zs = atom_loop.get("_atom_site_fract_z") or []
    occs = atom_loop.get("_atom_site_occupancy") or ["1"] * n
    grouped: dict[tuple[int, int, int], tuple[str, float, list[float]]] = {}
    for i in range(min(n, len(symbols), len(xs), len(ys), len(zs))):
        symbol = _normalize_symbol(re.sub(r"[0-9].*$", "", str(symbols[i])))
        if not symbol or not symbol[0].isalpha():
            continue
        fx, fy, fz = _finite(xs[i]), _finite(ys[i]), _finite(zs[i])
        occ = _finite(occs[i]) if i < len(occs) else 1.0
        if fx is None or fy is None or fz is None or occ is None or occ < 0.20:
            continue
        key = (int(round(fx * 100)), int(round(fy * 100)), int(round(fz * 100)))
        prev = grouped.get(key)
        if prev is None or occ > prev[1]:
            grouped[key] = (symbol, occ, [fx % 1.0, fy % 1.0, fz % 1.0])
    if len(grouped) < 2:
        raise ValueError("too few atom sites")
    species = [row[0] for row in grouped.values()]
    coords = [row[2] for row in grouped.values()]
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    hm = items.get("_symmetry_space_group_name_h-m") or items.get("_space_group_name_h-m_alt")
    if hm:
        hm = str(hm).strip().strip("'\"")
        try:
            return Structure.from_spacegroup(hm, lattice, species, coords)
        except Exception:
            pass
    return Structure(lattice, species, coords, coords_are_cartesian=False)


def structure_from_cif(text: str) -> Any:
    from pymatgen.core import Structure

    structure = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            structure = Structure.from_str(text, fmt="cif")
        except Exception:
            structure = None
    if structure is None:
        structure = structure_from_loops(text)
    species = [site_symbol(site) for site in structure]
    from pymatgen.core import Structure as PmgStructure

    rebuilt = PmgStructure(
        structure.lattice,
        species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )
    # Keep the published cell. Aggressive primitive standardization can
    # collapse hydrates / disordered experimental CIFs.
    try:
        return rebuilt.get_reduced_structure()
    except Exception:
        return rebuilt


def xrf_from_atoms(atom_types: list[str]) -> dict[str, float]:
    from pymatgen.core import Element

    masses: Counter[str] = Counter()
    for symbol in atom_types:
        try:
            masses[symbol] += float(Element(symbol).atomic_mass)
        except Exception:
            continue
    total = float(sum(masses.values()))
    if total <= 0:
        return {}
    return {symbol: mass / total for symbol, mass in sorted(masses.items())}


def cosine_on_grid(x_a, y_a, x_b, y_b) -> float | None:
    from crystal_language.spectra_encoder import rasterize_spectrum

    if not x_a or not y_a or not x_b or not y_b:
        return None
    va = rasterize_spectrum(x_a, y_a, GRID_MIN, GRID_MAX, GRID_STEP).numpy()
    vb = rasterize_spectrum(x_b, y_b, GRID_MIN, GRID_MAX, GRID_STEP).numpy()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0 or nb <= 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def to_native_record(
    *,
    cod_id: str,
    row: dict[str, str],
    structure: Any,
    peak_x: list[float],
    peak_y: list[float],
    pattern_meta: dict[str, Any],
    n_profile: int,
) -> dict[str, Any]:
    from pymatgen.core import Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from src.multimodal_utils import compute_neutron_pattern, compute_valence
    from src.structure_utils import compute_pxrd

    atom_types = [site_symbol(site) for site in structure]
    frac_coords = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    lattice = Lattice(lattice_matrix)
    pxrd_x, pxrd_y = compute_pxrd(structure, GRID_MIN, GRID_MAX)
    sim_nd = compute_neutron_pattern(
        structure,
        GRID_MIN,
        GRID_MAX,
        wavelength=float(pattern_meta.get("wavelength") or REF_WAVELENGTH),
    )
    valence = compute_valence(structure)
    try:
        space_group = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
    except Exception:
        space_group = int(_finite(row.get("sgNumber")) or 0)

    sim_nd_x = sim_nd[0].tolist() if sim_nd is not None else []
    sim_nd_y = (sim_nd[1] / max(float(np.max(sim_nd[1])), 1e-12)).tolist() if sim_nd is not None else []
    nd_cos = cosine_on_grid(peak_x, peak_y, sim_nd_x, sim_nd_y)

    pxrd_y = (pxrd_y / max(float(np.max(pxrd_y)), 1e-12)).tolist()
    pxrd_x = pxrd_x.tolist()

    return {
        "id": f"cod:{cod_id}",
        "source": "cod_experimental_npd",
        "source_id": str(cod_id),
        "split": "test",
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
            "pxrd_x": [round(float(v), 3) for v in pxrd_x],
            "pxrd_y": [round(float(v), 3) for v in pxrd_y],
            "neutron_x": [round(float(v), 3) for v in peak_x],
            "neutron_y": [round(float(v), 3) for v in peak_y],
            "xrf_mass_frac": xrf_from_atoms(atom_types),
            "valence_assignments": valence.get("valence_assignments") or {},
            "valence_source": valence.get("valence_source"),
            "valence_assigned": bool(valence.get("valence_assigned")),
        },
        "properties": {
            "formation_energy_per_atom": None,
            "e_above_hull": None,
            "band_gap": None,
        },
        "experimental": {
            "doi": row.get("doi") or None,
            "journal": row.get("journal") or None,
            "year": int(_finite(row.get("year")) or 0) or None,
            "title": row.get("title") or None,
            "wavelength_A": pattern_meta.get("wavelength"),
            "pattern_kind": pattern_meta.get("kind"),
            "n_profile_points": n_profile,
            "n_peaks": len(peak_x),
            "simulated_nd_cosine": nd_cos,
            "catalog_formula": row.get("formula") or None,
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def plot_examples(records: list[dict[str, Any]], out_dir: Path, n: int = 6) -> None:
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
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.2 * len(chosen)), sharex=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, rec in zip(axes, chosen):
        spec = rec["conditions"]
        ax.vlines(spec["neutron_x"], 0.0, spec["neutron_y"], color="#1f77b4", linewidth=1.2, label="exp. ND peaks")
        ax.vlines(spec["pxrd_x"], 0.0, spec["pxrd_y"], color="#d62728", linewidth=0.8, alpha=0.55, label="sim. PXRD")
        meta = rec["experimental"]
        ax.set_ylabel("I (rel.)")
        ax.set_xlim(GRID_MIN, GRID_MAX)
        ax.set_title(
            f"{rec['id']}  {rec['formula']}  λ={meta.get('wavelength_A')}  "
            f"cos={None if meta.get('simulated_nd_cosine') is None else round(meta['simulated_nd_cosine'], 3)}",
            loc="left",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("2θ (deg)")
    fig.tight_layout()
    fig.savefig(out_dir / "example_patterns.png", dpi=160)
    plt.close(fig)


def process_entry(row: dict[str, str], raw_dir: Path, max_sites: int = MAX_SITES) -> dict[str, Any]:
    cod_id = str(row["file"]).zfill(7)
    cif_path = raw_dir / f"{cod_id}.cif"
    hkl_path = raw_dir / f"{cod_id}.hkl"
    status: dict[str, Any] = {
        "id": cod_id,
        "status": "failed",
        "reason": None,
        "formula": row.get("formula"),
        "year": row.get("year"),
        "doi": row.get("doi"),
    }
    if not download(f"{COD_HTTP}/{cod_id}.cif", cif_path):
        status["reason"] = "cif_download_failed"
        return status
    download(f"{COD_HTTP}/{cod_id}.hkl", hkl_path)
    texts = [cif_path.read_text(encoding="utf-8", errors="replace")]
    if hkl_path.exists() and hkl_path.stat().st_size > 0:
        texts.append(hkl_path.read_text(encoding="utf-8", errors="replace"))
    catalog_wavelength = _parse_wavelength(row.get("wavelength"))
    pattern = extract_experimental_pattern(texts, catalog_wavelength)
    if pattern is None:
        status["reason"] = "no_parseable_pattern"
        return status
    x_clip, y_clip, peak_x, peak_y = clip_and_peaks(pattern["x"], pattern["y"])
    if len(peak_x) < MIN_PEAKS:
        status["reason"] = f"too_few_peaks:{len(peak_x)}"
        status["n_profile"] = len(x_clip)
        return status
    try:
        structure = structure_from_cif(texts[0])
    except Exception as exc:
        status["reason"] = f"structure_parse_failed:{type(exc).__name__}"
        return status
    n_sites = len(structure)
    n_elem = len({site_symbol(s) for s in structure})
    if n_sites < 2 or n_elem < 1:
        status["reason"] = f"dummy_or_empty:n_sites={n_sites}"
        return status
    if n_sites > max_sites:
        status["reason"] = f"too_many_sites:{n_sites}"
        status["n_sites"] = n_sites
        return status
    model_elems = {site_symbol(s) for s in structure}
    catalog_elems = set()
    for raw in (row.get("calcformula"), row.get("formula")):
        if not raw:
            continue
        catalog_elems.update(re.findall(r"[A-Z][a-z]?", raw.replace("D", "H")))
    missing = {el for el in catalog_elems if el not in model_elems and el not in {"D"}}
    if missing:
        status["reason"] = f"occupancy_collapsed:{sorted(missing)}"
        status["n_sites"] = n_sites
        return status
    native = to_native_record(
        cod_id=cod_id,
        row=row,
        structure=structure,
        peak_x=peak_x,
        peak_y=peak_y,
        pattern_meta=pattern,
        n_profile=len(x_clip),
    )
    status.update(
        {
            "status": "ok",
            "formula": native["formula"],
            "n_sites": native["n_sites"],
            "n_peaks": len(peak_x),
            "n_profile": len(x_clip),
            "pattern_kind": pattern["kind"],
            "wavelength_A": pattern["wavelength"],
            "simulated_nd_cosine": native["experimental"]["simulated_nd_cosine"],
            "native": native,
        }
    )
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=MAX_SITES)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Querying COD neutron-powder + Fobs catalog …")
    catalog = query_cod_catalog()
    if args.max_entries is not None:
        catalog = catalog[: args.max_entries]
    with (out_dir / "catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog[0].keys()))
        writer.writeheader()
        writer.writerows(catalog)
    print(f"Catalog: {len(catalog)} COD entries")

    def _prefetch(row: dict[str, str]) -> None:
        cod_id = str(row["file"]).zfill(7)
        download(f"{COD_HTTP}/{cod_id}.cif", raw_dir / f"{cod_id}.cif")
        download(f"{COD_HTTP}/{cod_id}.hkl", raw_dir / f"{cod_id}.hkl")

    from concurrent.futures import ThreadPoolExecutor

    print("Downloading CIF/HKL …")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_prefetch, catalog))

    results: list[dict[str, Any]] = []
    for i, row in enumerate(catalog, start=1):
        print(f"[{i}/{len(catalog)}] COD {row['file']} …", flush=True)
        results.append(process_entry(row, raw_dir, max_sites=int(args.max_sites)))

    ok_raw = [r for r in results if r.get("status") == "ok"]
    for row in ok_raw:
        native = row["native"]
        n_elem = len(set(native["atom_types"]))
        cos = native["experimental"].get("simulated_nd_cosine")
        if n_elem < 2:
            row["status"] = "failed"
            row["reason"] = "dummy_element"
            row.pop("native", None)
            continue
        if cos is not None and float(cos) < 0.15:
            row["status"] = "failed"
            row["reason"] = f"nd_cosine_low:{cos:.3f}"
            row.pop("native", None)

    ok = [r for r in results if r.get("status") == "ok"]
    native_records = [r["native"] for r in ok]
    write_jsonl(out_dir / "native" / "test.jsonl", native_records)

    from crystal_language.data.sequence import build_sequence_record

    seq_records = []
    seq_failed = 0
    for record in native_records:
        try:
            seq_records.append(build_sequence_record(record, rep="v2"))
        except Exception:
            seq_failed += 1
    write_jsonl(out_dir / "crystal_language_v2" / "test.jsonl", seq_records)
    plot_examples(native_records, fig_dir)

    skip_counts: Counter[str] = Counter()
    for row in results:
        if row.get("status") != "ok":
            skip_counts[str(row.get("reason") or "unknown")] += 1

    summary = {
        "catalog": len(catalog),
        "usable": len(native_records),
        "sequence_records": len(seq_records),
        "sequence_failed": seq_failed,
        "skip_reasons": dict(skip_counts),
        "max_sites": int(args.max_sites),
        "grid_2theta": [GRID_MIN, GRID_MAX],
        "notes": [
            "neutron_x/y are experimental (peak-picked onto 5-80 deg)",
            "pxrd_x/y are simulated from the COD CIF",
            "D sites mapped to H in the LM target",
            "TOF / d-spacing patterns converted with experimental λ when known, else 1.54184 Å",
        ],
        "eval_example": (
            "python crystal_language/evaluate_generation.py "
            f"--data-dir {out_dir / 'crystal_language_v2'} "
            "--split test --condition-set pxrd_neutron --rerank"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_fields = [
        "id",
        "status",
        "reason",
        "formula",
        "n_sites",
        "n_peaks",
        "n_profile",
        "pattern_kind",
        "wavelength_A",
        "simulated_nd_cosine",
        "year",
        "doi",
    ]
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(json.dumps(summary, indent=2))
    return 0 if native_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
