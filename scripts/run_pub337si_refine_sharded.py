#!/usr/bin/env python3
"""Sharded refine-before-match for the public-337 SI ladder.

Per rung: CHGNet scoring (prerequisite), then one refinement job per shard
file and per backend (CHGNet, MACE) so the heavy formula-conditioned rungs
spread over every leftover GPU instead of running as one long chain. Existing
outputs are skipped, so this can be restarted or run alongside finished chains.
Finally merges both passes into <pool>_refined_cm and prints the ladder report.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
PY = "/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python"
EV = ROOT / "crystal_language/eval_results"
CH = EV / "chgnet"
LADDER = EV / "ladder_public_si"
NATIVE = ROOT / "data_prep/outputs/public_exp_pxrd_si/native/test.jsonl"
CONDS = ["pxrd_formula", "pxrd_neutron_xrf", "pxrd_xrf", "pxrd", "pxrd_neutron", "pxrd_neutron_xrf_xps"]
GLOB = "test_*_s100.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))
from run_rps_si_di_local import pick_gpu  # noqa: E402  leftover rule: util<=15%, >=12 GiB free


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def energy_path(cond: str, lat: int) -> Path:
    return CH / f"ladder_public_si_{cond}_lat{lat}_chgnet.jsonl"


def score_cmd(cond: str, lat: int) -> list[str]:
    pool = LADDER / f"{cond}_lat{lat}"
    return [PY, "-u", str(ROOT / "scripts/score_candidates_mlip.py"), "--eval-dir", str(pool), "--out", str(energy_path(cond, lat)),
            "--backend", "chgnet", "--cond", cond, "--glob", GLOB, "--resume"]


def refine_cmd(cond: str, lat: int, backend: str, shard_file: str) -> list[str]:
    pool = LADDER / f"{cond}_lat{lat}"
    out = f"{pool}_f20" if backend == "chgnet" else f"{pool}_f20_mace"
    return [PY, "-u", str(ROOT / "scripts/refine_match_candidates.py"), "--eval-dir", str(pool), "--native", str(NATIVE),
            "--out-dir", out, "--rank-by", "fusion", "--energy", str(energy_path(cond, lat)), "--top", "20",
            "--glob", shard_file, "--backend", backend]


def refine_done(cond: str, lat: int, backend: str, shard_file: str) -> bool:
    pool = LADDER / f"{cond}_lat{lat}"
    out = Path(f"{pool}_f20" if backend == "chgnet" else f"{pool}_f20_mace")
    return (out / shard_file).exists()


def launch(cmd: list[str], gpu: int, tag: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "3"})
    logf = (ROOT / f"logs/pub337si_shard_{tag}_gpu{gpu}.log").open("w")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--per-gpu", type=int, default=2, help="refinement processes allowed per GPU (they are CPU-bound)")
    ap.add_argument("--lat", default="1,0")
    ap.add_argument("--backends", default="chgnet,mace")
    args = ap.parse_args()
    lats = [int(x) for x in args.lat.split(",")]
    backends = [b for b in args.backends.split(",") if b]
    rungs = [(c, lat) for lat in lats for c in CONDS if (LADDER / f"{c}_lat{lat}").exists()]

    # 1. scoring prerequisites (fast; sequential on one leftover GPU)
    log("STAGE scoring")
    for cond, lat in rungs:
        if energy_path(cond, lat).exists():
            continue
        gpu = None
        while gpu is None:
            gpu = pick_gpu(set())
            if gpu is None:
                time.sleep(60)
        log(f"score {cond}_lat{lat} gpu={gpu}")
        subprocess.run(score_cmd(cond, lat), cwd=str(ROOT), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                       stdout=(ROOT / f"logs/pub337si_score_{cond}_lat{lat}.log").open("w"), stderr=subprocess.STDOUT, check=False)

    # 2. sharded refinement jobs
    jobs: list[tuple[str, int, str, str]] = []
    for cond, lat in rungs:
        pool = LADDER / f"{cond}_lat{lat}"
        for backend in backends:
            for p in sorted(pool.glob(GLOB)):
                if not refine_done(cond, lat, backend, p.name):
                    jobs.append((cond, lat, backend, p.name))
    log(f"STAGE refine  jobs={len(jobs)}")
    running: dict[int, tuple[tuple, subprocess.Popen, int]] = {}  # pid -> (job, proc, gpu)
    failed = 0
    while jobs or running:
        for pid in list(running):
            job, proc, gpu = running[pid]
            code = proc.poll()
            if code is None:
                continue
            log(f"{'DONE' if code == 0 else 'FAIL'} {job[0]}_lat{job[1]} {job[2]} {job[3]} gpu={gpu} exit={code}")
            failed += int(code != 0)
            del running[pid]
        while jobs and len(running) < args.max_concurrent:
            load = {}
            for _, _, gpu in running.values():
                load[gpu] = load.get(gpu, 0) + 1
            busy = {g for g, n in load.items() if n >= args.per_gpu}
            gpu = pick_gpu(busy)
            if gpu is None:
                break
            job = jobs.pop(0)
            tag = f"{job[0]}_lat{job[1]}_{job[2]}_{job[3].split('_start')[1].split('_n')[0]}"
            proc = launch(refine_cmd(*job), gpu, tag)
            running[proc.pid] = (job, proc, gpu)
            log(f"START {job[0]}_lat{job[1]} {job[2]} {job[3]} gpu={gpu} remaining={len(jobs)}")
            time.sleep(3)
        time.sleep(30)
    log(f"refine finished failed={failed}")

    # 3. merge + report
    log("STAGE merge")
    for cond, lat in rungs:
        pool = LADDER / f"{cond}_lat{lat}"
        base, extra, out = Path(f"{pool}_f20"), Path(f"{pool}_f20_mace"), Path(f"{pool}_refined_cm")
        if base.exists() and (extra.exists() or "mace" not in backends):
            cmd = [PY, str(ROOT / "scripts/merge_refined_pools.py"), "--base", str(base), "--out", str(out)]
            cmd += ["--extra", str(extra)] if extra.exists() else ["--extra", str(base)]
            subprocess.run(cmd, check=False)
    log("STAGE report")
    subprocess.run([PY, str(ROOT / "scripts/ladder_public_si_report.py"), "--out", str(CH / "ladder_public_si_report.json")], check=False)
    log("DONE public-337 SI sharded refinement")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
