#!/usr/bin/env python3
"""Score the RPS-specialist SI evals with CHGNet/MACE and print vs RPS.

Waits for generation shards, then energy-scores CNRS then RRUFF.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
PY = Path("/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python")
PIDFILE = ROOT / "logs/rps_lat_eval.pid"
SCORE = ROOT / "scripts/score_candidates_mlip.py"
ANALYZE = ROOT / "scripts/analyze_rps_step1.py"
EVAL = ROOT / "crystal_language/eval_results"
CHG = EVAL / "chgnet"

EXPECT = [
    (EVAL / "rps_cnrs126_si_profile_rpslat", 126, 32, 100),
    (EVAL / "rps_rruff269_si_peaks_rpslat", 269, 32, 100),
]

JOBS = [
    {
        "name": "CNRS RPS-lat CHGNet N=100 T=1.0",
        "rps": "cnrs",
        "base_eval": EVAL / "rps_cnrs126_si_profile_rpslat",
        "base_energy": CHG / "rps_cnrs126_si_profile_rpslat_chgnet.jsonl",
        "backend": "chgnet",
    },
    {
        "name": "CNRS RPS-lat MACE N=100 T=1.0",
        "rps": "cnrs",
        "base_eval": EVAL / "rps_cnrs126_si_profile_rpslat",
        "base_energy": CHG / "rps_cnrs126_si_profile_rpslat_mace.jsonl",
        "backend": "mace",
    },
    {
        "name": "RRUFF RPS-lat CHGNet N=100 T=1.0",
        "rps": "rruff",
        "base_eval": EVAL / "rps_rruff269_si_peaks_rpslat",
        "base_energy": CHG / "rps_rruff269_si_peaks_rpslat_chgnet.jsonl",
        "backend": "chgnet",
    },
    {
        "name": "RRUFF RPS-lat MACE N=100 T=1.0",
        "rps": "rruff",
        "base_eval": EVAL / "rps_rruff269_si_peaks_rpslat",
        "base_energy": CHG / "rps_rruff269_si_peaks_rpslat_mace.jsonl",
        "backend": "mace",
    },
]


def ready() -> bool:
    if PIDFILE.exists():
        return False
    for out_dir, n, shard, nsamp in EXPECT:
        for start in range(0, n, shard):
            summary = out_dir / (
                f"test_pxrd_formula_lat1_full_inject0_cd1_rr1"
                f"_start{start}_n{shard}_s{nsamp}_summary.json"
            )
            jsonl = summary.with_name(summary.name.replace("_summary.json", ".jsonl"))
            if not (summary.exists() and jsonl.exists() and jsonl.stat().st_size > 0):
                return False
    return True


def score(eval_dir: Path, out: Path, backend: str) -> None:
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP score exists {out}", flush=True)
        return
    cmd = [
        str(PY),
        "-u",
        str(SCORE),
        "--eval-dir",
        str(eval_dir),
        "--out",
        str(out),
        "--backend",
        backend,
        "--glob",
        "test_pxrd_formula_lat1_*_s100.jsonl",
        "--resume",
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def analyze(job: dict) -> None:
    cmd = [
        str(PY),
        str(ANALYZE),
        "--name",
        job["name"],
        "--base-eval",
        str(job["base_eval"]),
        "--base-energy",
        str(job["base_energy"]),
        "--rps",
        job["rps"],
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    CHG.mkdir(parents=True, exist_ok=True)
    print("waiting for RPS-lat generation shards...", flush=True)
    while not ready():
        time.sleep(30)
    print("generation ready", flush=True)
    for job in JOBS:
        score(job["base_eval"], job["base_energy"], job["backend"])
        analyze(job)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
