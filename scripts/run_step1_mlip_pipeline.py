#!/usr/bin/env python3
"""After adaptive-N generation: score extra pools and print Step-1 vs RPS.

Runs on a single leftover GPU. Waits for the adapt dispatcher pidfile to vanish
and for all expected shard summaries to exist, then scores CHGNet + MACE on the
new candidates and fuses with the honest T=1.0 N=100 pools.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
PY = Path("/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python")
PIDFILE = ROOT / "logs/rps_adapt_n400.pid"
SCORE = ROOT / "scripts/score_candidates_mlip.py"
ANALYZE = ROOT / "scripts/analyze_rps_step1.py"
EVAL = ROOT / "crystal_language/eval_results"
CHG = EVAL / "chgnet"

JOBS = [
    {
        "name": "CNRS T=1.0 + MACE (N=100 only)",
        "rps": "cnrs",
        "base_eval": EVAL / "rps_cnrs126_si_profile_sweep_t10k50",
        "base_energy": CHG / "rps_cnrs126_si_profile_sweep_t10k50_mace.jsonl",
        "extra_eval": "",
        "extra_energy": "",
        "need_score_base_mace": True,
        "need_score_extra": False,
    },
    {
        "name": "RRUFF T=1.0 + MACE (N=100 only)",
        "rps": "rruff",
        "base_eval": EVAL / "rps_rruff269_si_peaks_sweep_t10k50",
        "base_energy": CHG / "rps_rruff269_si_peaks_sweep_t10k50_mace.jsonl",
        "extra_eval": "",
        "extra_energy": "",
        "need_score_base_mace": True,
        "need_score_extra": False,
    },
    {
        "name": "CNRS adaptive N=500 (100+400) CHGNet",
        "rps": "cnrs",
        "base_eval": EVAL / "rps_cnrs126_si_profile_sweep_t10k50",
        "base_energy": CHG / "rps_cnrs126_si_profile_sweep_t10k50.jsonl",
        "extra_eval": EVAL / "rps_cnrs126_si_adapt_n400",
        "extra_energy": CHG / "rps_cnrs126_si_adapt_n400_chgnet.jsonl",
        "need_score_base_mace": False,
        "need_score_extra": True,
        "extra_backend": "chgnet",
        "wait_adapt": True,
    },
    {
        "name": "CNRS adaptive N=500 (100+400) MACE",
        "rps": "cnrs",
        "base_eval": EVAL / "rps_cnrs126_si_profile_sweep_t10k50",
        "base_energy": CHG / "rps_cnrs126_si_profile_sweep_t10k50_mace.jsonl",
        "extra_eval": EVAL / "rps_cnrs126_si_adapt_n400",
        "extra_energy": CHG / "rps_cnrs126_si_adapt_n400_mace.jsonl",
        "need_score_base_mace": False,
        "need_score_extra": True,
        "extra_backend": "mace",
        "wait_adapt": True,
    },
    {
        "name": "RRUFF adaptive N=500 (100+400) CHGNet",
        "rps": "rruff",
        "base_eval": EVAL / "rps_rruff269_si_peaks_sweep_t10k50",
        "base_energy": CHG / "rps_rruff269_si_peaks_sweep_t10k50.jsonl",
        "extra_eval": EVAL / "rps_rruff269_si_adapt_n400",
        "extra_energy": CHG / "rps_rruff269_si_adapt_n400_chgnet.jsonl",
        "need_score_base_mace": False,
        "need_score_extra": True,
        "extra_backend": "chgnet",
        "wait_adapt": True,
    },
    {
        "name": "RRUFF adaptive N=500 (100+400) MACE",
        "rps": "rruff",
        "base_eval": EVAL / "rps_rruff269_si_peaks_sweep_t10k50",
        "base_energy": CHG / "rps_rruff269_si_peaks_sweep_t10k50_mace.jsonl",
        "extra_eval": EVAL / "rps_rruff269_si_adapt_n400",
        "extra_energy": CHG / "rps_rruff269_si_adapt_n400_mace.jsonl",
        "need_score_base_mace": False,
        "need_score_extra": True,
        "extra_backend": "mace",
        "wait_adapt": True,
    },
]

ADAPT_EXPECT = [
    (EVAL / "rps_cnrs126_si_adapt_n400", 21, 8, 400),
    (EVAL / "rps_rruff269_si_adapt_n400", 117, 8, 400),
]


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def score(eval_dir: Path, out: Path, backend: str) -> None:
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP score exists {out}", flush=True)
        return
    run(
        [
            str(PY),
            "-u",
            str(SCORE),
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--backend",
            backend,
            "--resume",
        ]
    )


def adapt_ready() -> bool:
    if PIDFILE.exists():
        return False
    for out_dir, n, shard, nsamp in ADAPT_EXPECT:
        starts = list(range(0, n, shard))
        for start in starts:
            summary = out_dir / (
                f"test_pxrd_formula_lat1_full_inject0_cd1_rr1"
                f"_start{start}_n{shard}_s{nsamp}_summary.json"
            )
            jsonl = summary.with_name(summary.name.replace("_summary.json", ".jsonl"))
            if not (summary.exists() and jsonl.exists() and jsonl.stat().st_size > 0):
                return False
    return True


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
    if job.get("extra_eval"):
        cmd += ["--extra-eval", str(job["extra_eval"]), "--extra-energy", str(job["extra_energy"])]
    run(cmd)


def main() -> int:
    CHG.mkdir(parents=True, exist_ok=True)
    # Immediate: MACE on the frozen T=1.0 N=100 pools (no generation wait).
    score(
        EVAL / "rps_cnrs126_si_profile_sweep_t10k50",
        CHG / "rps_cnrs126_si_profile_sweep_t10k50_mace.jsonl",
        "mace",
    )
    analyze(JOBS[0])
    score(
        EVAL / "rps_rruff269_si_peaks_sweep_t10k50",
        CHG / "rps_rruff269_si_peaks_sweep_t10k50_mace.jsonl",
        "mace",
    )
    analyze(JOBS[1])

    print("waiting for adaptive-N shards...", flush=True)
    while not adapt_ready():
        time.sleep(30)
    print("adaptive-N ready", flush=True)

    score(EVAL / "rps_cnrs126_si_adapt_n400", CHG / "rps_cnrs126_si_adapt_n400_chgnet.jsonl", "chgnet")
    score(EVAL / "rps_cnrs126_si_adapt_n400", CHG / "rps_cnrs126_si_adapt_n400_mace.jsonl", "mace")
    score(EVAL / "rps_rruff269_si_adapt_n400", CHG / "rps_rruff269_si_adapt_n400_chgnet.jsonl", "chgnet")
    score(EVAL / "rps_rruff269_si_adapt_n400", CHG / "rps_rruff269_si_adapt_n400_mace.jsonl", "mace")

    for job in JOBS[2:]:
        analyze(job)

    out = CHG / "rps_step1_summary.log"
    print(f"done  see dispatcher log + {out}", flush=True)
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
