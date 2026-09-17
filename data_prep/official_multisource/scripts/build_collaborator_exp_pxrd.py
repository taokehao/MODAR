#!/usr/bin/env python
"""Build an eval set from collaborator experimental PXRD + CIF.

Each sample is a Bruker RAW1.01 powder pattern paired with a CIF. The PXRD
condition is the measured profile, Q-aligned onto the training 2θ grid
(λ=1.54184 Å, 5–80°). Neutron / XRF / XPS are simulated from the CIF so this
set can test the value of *simulated* neutron given *experimental* PXRD.

Usage (LLMD env)::

  python data_prep/official_multisource/scripts/build_collaborator_exp_pxrd.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

GRID_MIN = 5.0
GRID_MAX = 80.0
GRID_STEP = 0.05
REF_WAVELENGTH = 1.54184
MIN_POINTS = 8

DEFAULT_XRD_DIR = PROJECT_ROOT / "XRD"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_prep" / "outputs" / "collaborator_exp_pxrd"

# CIF stem -> RAW stem when filenames do not match 1:1.
RAW_ALIASES = {
    "Cu.1.38Mo3S4": "Cu2Mo3S4",
}


def _c_string(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("latin1", errors="replace").strip()


def q_from_two_theta(two_theta_deg: float, wavelength: float) -> float | None:
    if wavelength <= 0:
        return None
    s = math.sin(math.radians(two_theta_deg) / 2.0)
    if s <= 0:
        return None
    return 4.0 * math.pi * s / wavelength


def two_theta_from_q(q: float, wavelength: float) -> float | None:
    arg = q * wavelength / (4.0 * math.pi)
    if not (0.0 < arg < 1.0):
        return None
    return math.degrees(2.0 * math.asin(arg))


def remap_two_theta_to_train(
    xs: np.ndarray,
    ys: np.ndarray,
    lambda_native: float,
    lambda_train: float = REF_WAVELENGTH,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if lambda_native is None or lambda_native <= 0 or abs(float(lambda_native) - lambda_train) < 0.01:
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


def encode_profile(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_min: float = GRID_MIN,
    grid_max: float = GRID_MAX,
) -> tuple[list[float], list[float], list[float], list[float]]:
    mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    xs, ys = xs[mask], ys[mask]
    mask = (xs >= grid_min) & (xs <= grid_max)
    xs, ys = xs[mask], ys[mask]
    if xs.size == 0:
        return [], [], [], []
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    ys = ys / float(np.max(ys))
    n = int(round((grid_max - grid_min) / GRID_STEP)) + 1
    acc = np.zeros(n, dtype=float)
    idx = np.round((xs - grid_min) / GRID_STEP).astype(int)
    keep = (idx >= 0) & (idx < n)
    for i, y in zip(idx[keep], ys[keep]):
        if y > acc[i]:
            acc[i] = float(y)
    bkg = _rolling_percentile(acc, window=max(21, n // 25), q=15.0)
    net = np.clip(acc - bkg, 0.0, None)
    peak = float(net.max())
    if peak <= 0:
        net = acc
        peak = float(net.max())
    if peak <= 0:
        return xs.tolist(), ys.tolist(), [], []
    net = net / peak
    grid = grid_min + np.arange(n) * GRID_STEP
    keep = net >= 0.02
    if int(keep.sum()) < MIN_POINTS:
        keep = net >= 0.005
    return xs.tolist(), ys.tolist(), grid[keep].tolist(), net[keep].tolist()


def read_bruker_raw101(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if not data.startswith(b"RAW1.01"):
        raise ValueError(f"{path.name}: not a Bruker RAW1.01 file")
    range_cnt = struct.unpack_from("<i", data, 12)[0]
    if range_cnt < 1:
        raise ValueError(f"{path.name}: no measured ranges")
    header = {
        "date": _c_string(data[16:26]),
        "time": _c_string(data[26:36]),
        "user": _c_string(data[36:108]),
        "site": _c_string(data[108:326]),
        "sample": _c_string(data[326:386]),
        "comment": _c_string(data[386:546]),
        "anode": _c_string(data[608:612]),
        "alpha_average": float(struct.unpack_from("<d", data, 616)[0]),
        "alpha1": float(struct.unpack_from("<d", data, 624)[0]),
        "alpha2": float(struct.unpack_from("<d", data, 632)[0]),
        "n_ranges": int(range_cnt),
    }
    offset = 712
    best: dict[str, Any] | None = None
    for _ in range(range_cnt):
        if offset + 304 > len(data):
            break
        header_len = struct.unpack_from("<i", data, offset)[0]
        steps = struct.unpack_from("<i", data, offset + 4)[0]
        start_2theta = struct.unpack_from("<d", data, offset + 16)[0]
        step_size = struct.unpack_from("<d", data, offset + 176)[0]
        used_lambda = struct.unpack_from("<d", data, offset + 240)[0]
        extra = struct.unpack_from("<i", data, offset + 256)[0]
        if header_len <= 0:
            header_len = 304
        data_off = offset + header_len + max(0, extra)
        nbytes = steps * 4
        if steps < MIN_POINTS or data_off + nbytes > len(data):
            offset = data_off + max(0, nbytes)
            continue
        y = np.frombuffer(data, dtype="<f4", count=steps, offset=data_off).astype(float)
        x = start_2theta + np.arange(steps, dtype=float) * step_size
        rec = {
            "x": x,
            "y": y,
            "start_2theta": float(start_2theta),
            "step": float(step_size),
            "n_points": int(steps),
            "wavelength_A": float(used_lambda) if 0.4 < used_lambda < 4.5 else None,
        }
        if best is None or rec["n_points"] > best["n_points"]:
            best = rec
        offset = data_off + nbytes
    if best is None:
        raise ValueError(f"{path.name}: could not read intensity block")
    if best["wavelength_A"] is None:
        avg = header["alpha_average"]
        best["wavelength_A"] = float(avg) if 0.4 < avg < 4.5 else REF_WAVELENGTH
    best["header"] = header
    return best


def pair_files(xrd_dir: Path) -> list[tuple[Path, Path, str]]:
    cifs = {p.stem: p for p in xrd_dir.glob("*.cif")}
    raws = {p.stem: p for p in xrd_dir.glob("*.raw")}
    pairs: list[tuple[Path, Path, str]] = []
    for stem, cif in sorted(cifs.items()):
        raw_stem = RAW_ALIASES.get(stem, stem)
        raw = raws.get(raw_stem)
        if raw is None:
            continue
        label = re.sub(r"[^A-Za-z0-9.+()-]+", "_", stem).strip("_")
        pairs.append((cif, raw, label))
    return pairs


def site_symbol(site: Any) -> str:
    if getattr(site, "is_ordered", False):
        symbol = site.specie.symbol
    else:
        species = getattr(site, "species", None)
        if species is None:
            symbol = str(site.specie)
        else:
            symbol = max(species.items(), key=lambda kv: kv[1])[0].symbol
    return symbol


def occupancy_weighted_symbols(structure: Any) -> list[str]:
    """Approximate composition by repeating majority species; XRF uses masses below."""
    return [site_symbol(site) for site in structure]


def occupancy_mass_frac(structure: Any) -> dict[str, float]:
    from pymatgen.core import Element

    masses: Counter[str] = Counter()
    for site in structure:
        if getattr(site, "is_ordered", False):
            try:
                masses[site.specie.symbol] += float(Element(site.specie.symbol).atomic_mass)
            except Exception:
                continue
            continue
        species = getattr(site, "species", None)
        if species is None:
            continue
        for spec, occ in species.items():
            try:
                masses[spec.symbol] += float(Element(spec.symbol).atomic_mass) * float(occ)
            except Exception:
                continue
    total = float(sum(masses.values()))
    if total <= 0:
        return {}
    return {sym: mass / total for sym, mass in sorted(masses.items())}


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


def load_structure(cif_path: Path) -> Any:
    from pymatgen.core import Structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = Structure.from_file(str(cif_path))
    species = [site_symbol(site) for site in structure]
    rebuilt = structure.__class__(
        structure.lattice,
        species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )
    try:
        rebuilt = rebuilt.get_reduced_structure()
    except Exception:
        pass
    return rebuilt, structure


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
    label: str,
    cif_name: str,
    raw_name: str,
    structure: Any,
    raw_structure: Any,
    peak_x: list[float],
    peak_y: list[float],
    pattern_meta: dict[str, Any],
    n_profile: int,
    q_aligned: bool,
) -> dict[str, Any]:
    from pymatgen.core import Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from src.multimodal_utils import compute_neutron_pattern, compute_valence
    from src.structure_utils import compute_pxrd

    atom_types = occupancy_weighted_symbols(structure)
    frac_coords = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    lattice = Lattice(lattice_matrix)
    sim_pxrd_x, sim_pxrd_y = compute_pxrd(structure, GRID_MIN, GRID_MAX)
    sim_nd = compute_neutron_pattern(structure, GRID_MIN, GRID_MAX, wavelength=REF_WAVELENGTH)
    valence = compute_valence(structure)
    try:
        space_group = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
    except Exception:
        space_group = 0

    sim_pxrd_y = (sim_pxrd_y / max(float(np.max(sim_pxrd_y)), 1e-12)).tolist()
    sim_pxrd_x = sim_pxrd_x.tolist()
    sim_nd_x = sim_nd[0].tolist() if sim_nd is not None else []
    sim_nd_y = (sim_nd[1] / max(float(np.max(sim_nd[1])), 1e-12)).tolist() if sim_nd is not None else []
    pxrd_cos = cosine_on_grid(peak_x, peak_y, sim_pxrd_x, sim_pxrd_y)
    disordered = any(not getattr(site, "is_ordered", True) for site in raw_structure)

    xrf = occupancy_mass_frac(raw_structure)
    if not xrf:
        xrf = xrf_from_atoms(atom_types)

    return {
        "id": f"collab:{label}",
        "source": "collaborator_exp_pxrd",
        "source_id": label,
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
            "pxrd_x": [round(float(v), 3) for v in peak_x],
            "pxrd_y": [round(float(v), 3) for v in peak_y],
            "neutron_x": [round(float(v), 3) for v in sim_nd_x],
            "neutron_y": [round(float(v), 3) for v in sim_nd_y],
            "xrf_mass_frac": xrf,
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
            "cif": cif_name,
            "raw": raw_name,
            "catalog_formula": raw_structure.composition.reduced_formula,
            "wavelength_A": pattern_meta.get("wavelength_A"),
            "anode": (pattern_meta.get("header") or {}).get("anode"),
            "scan_start": pattern_meta.get("start_2theta"),
            "scan_step": pattern_meta.get("step"),
            "n_raw_points": pattern_meta.get("n_points"),
            "n_profile_points": n_profile,
            "n_pxrd_bins": len(peak_x),
            "q_aligned": bool(q_aligned),
            "disordered_cif": bool(disordered),
            "simulated_pxrd_cosine": pxrd_cos,
            "date": (pattern_meta.get("header") or {}).get("date"),
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


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
        ax.vlines(spec["neutron_x"], 0.0, spec["neutron_y"], color="#1f77b4", linewidth=0.8, alpha=0.65, label="sim. ND")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xrd-dir", default=str(DEFAULT_XRD_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    xrd_dir = Path(args.xrd_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_files(xrd_dir)
    results: list[dict[str, Any]] = []
    for cif_path, raw_path, label in pairs:
        row: dict[str, Any] = {
            "id": label,
            "cif": cif_path.name,
            "raw": raw_path.name,
            "status": "failed",
        }
        print(f"{label}: {cif_path.name} + {raw_path.name}", flush=True)
        try:
            pattern = read_bruker_raw101(raw_path)
        except Exception as exc:
            row["reason"] = f"raw_parse_failed:{type(exc).__name__}:{exc}"
            results.append(row)
            continue
        xs, ys, q_aligned = remap_two_theta_to_train(
            pattern["x"], pattern["y"], float(pattern["wavelength_A"] or REF_WAVELENGTH)
        )
        x_clip, _, peak_x, peak_y = encode_profile(xs, ys)
        n_profile = len(x_clip)
        if len(peak_x) < MIN_POINTS:
            row["reason"] = f"too_few_pxrd_bins:{len(peak_x)}"
            results.append(row)
            continue
        try:
            structure, raw_structure = load_structure(cif_path)
        except Exception as exc:
            row["reason"] = f"cif_parse_failed:{type(exc).__name__}:{exc}"
            results.append(row)
            continue
        native = to_native_record(
            label=label,
            cif_name=cif_path.name,
            raw_name=raw_path.name,
            structure=structure,
            raw_structure=raw_structure,
            peak_x=peak_x,
            peak_y=peak_y,
            pattern_meta=pattern,
            n_profile=n_profile,
            q_aligned=q_aligned,
        )
        row.update(
            {
                "status": "ok",
                "formula": native["formula"],
                "catalog_formula": native["experimental"]["catalog_formula"],
                "n_sites": native["n_sites"],
                "space_group": native["space_group"],
                "n_pxrd_bins": len(peak_x),
                "wavelength_A": native["experimental"]["wavelength_A"],
                "anode": native["experimental"]["anode"],
                "q_aligned": q_aligned,
                "disordered_cif": native["experimental"]["disordered_cif"],
                "simulated_pxrd_cosine": native["experimental"]["simulated_pxrd_cosine"],
                "native": native,
            }
        )
        results.append(row)

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
        "n_pairs": len(pairs),
        "usable": len(native_records),
        "sequence_records": len(seq_records),
        "sequence_failed": seq_failed,
        "skip_reasons": dict(skip_counts),
        "notes": [
            "pxrd_x/y are experimental Bruker RAW profiles, Q-aligned to λ=1.54184 Å and rasterized onto 5-80°",
            "neutron_x/y are simulated from the CIF (pymatgen NDCalculator, same wavelength/grid as training)",
            "XRF uses occupancy-weighted CIF composition; LM target uses majority species per site",
        ],
        "eval_example": (
            "python crystal_language/evaluate_generation.py "
            f"--data-dir {out_dir / 'crystal_language_v2'} "
            "--split test --condition-set pxrd_neutron_xrf --rerank"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "id",
        "status",
        "reason",
        "formula",
        "catalog_formula",
        "n_sites",
        "space_group",
        "n_pxrd_bins",
        "wavelength_A",
        "anode",
        "q_aligned",
        "disordered_cif",
        "simulated_pxrd_cosine",
        "cif",
        "raw",
    ]
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(json.dumps(summary, indent=2))
    return 0 if native_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
