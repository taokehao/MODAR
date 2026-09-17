#!/usr/bin/env python3
"""Post-generation pipeline for the public-337 SI ladder.

Waits for the shard dispatcher to finish, then for every rung (12 pools):
  1. CHGNet single-point energies (fusion ranker input)
  2. refine-before-match of the fusion top-20: CHGNet, then MACE
  3. merge the two passes into <pool>_refined_cm
Rungs are processed as independent chains, one per GPU from --gpus, known-lattice
rungs first. Finally prints the ladder report with bootstrap CIs.
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
DISPATCH_LOG = ROOT / "logs/pub337si_dispatcher.log"
CONDS = ["pxrd_formula", "pxrd_neutron_xrf", "pxrd_xrf", "pxrd", "pxrd_neutron", "pxrd_neutron_xrf_xps"]
GLOB = "test_*_s100.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def shards_done(pool: Path, n_targets: int = 337, shard: int = 32) -> bool:
    n_shards = (n_targets + shard - 1) // shard
    return len(list(pool.glob("test_*_s100_summary.json"))) >= n_shards and len(list(pool.glob(GLOB))) >= n_shards


def chain_cmds(cond: str, lat: int) -> list[list[str]]:
    pool = LADDER / f"{cond}_lat{lat}"
    energy = CH / f"ladder_public_si_{cond}_lat{lat}_chgnet.jsonl"
    cmds = [
        [PY, "-u", str(ROOT / "scripts/score_candidates_mlip.py"), "--eval-dir", str(pool), "--out", str(energy),
         "--backend", "chgnet", "--cond", cond, "--glob", GLOB, "--resume"],
        [PY, "-u", str(ROOT / "scripts/refine_match_candidates.py"), "--eval-dir", str(pool), "--native", str(NATIVE),
         "--out-dir", f"{pool}_f20", "--rank-by", "fusion", "--energy", str(energy), "--top", "20", "--glob", GLOB],
        [PY, "-u", str(ROOT / "scripts/refine_match_candidates.py"), "--eval-dir", str(pool), "--native", str(NATIVE),
         "--out-dir", f"{pool}_f20_mace", "--rank-by", "fusion", "--energy", str(energy), "--top", "20", "--glob", GLOB,
         "--backend", "mace"],
        [PY, str(ROOT / "scripts/merge_refined_pools.py"), "--base", f"{pool}_f20", "--extra", f"{pool}_f20_mace",
         "--out", f"{pool}_refined_cm"],
    ]
    return cmds


def run_chain(cond: str, lat: int, gpu: int) -> subprocess.Popen:
    script = " && ".join(" ".join(f"'{a}'" for a in cmd) for cmd in chain_cmds(cond, lat))
    logf = (ROOT / f"logs/pub337si_post_{cond}_lat{lat}_gpu{gpu}.log").open("w")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"})
    return subprocess.Popen(["bash", "-c", script], cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=3, help="chains running at once (one leftover GPU each)")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_rps_si_di_local import pick_gpu  # leftover rule: util<=15%, >=12 GiB free

    jobs = [(c, lat) for lat in (1, 0) for c in CONDS]

    log("STAGE wait for generation shards")
    while True:
        ready = [j for j in jobs if shards_done(LADDER / f"{j[0]}_lat{j[1]}")]
        finished = DISPATCH_LOG.exists() and "all shards finished" in DISPATCH_LOG.read_text()
        if len(ready) == len(jobs) or finished or args.no_wait:
            break
        time.sleep(120)
    log(f"generation: {len(ready)}/{len(jobs)} rungs complete")

    log("STAGE score + refine (per-rung chains)")
    pending = [j for j in jobs if shards_done(LADDER / f"{j[0]}_lat{j[1]}") and not (LADDER / f"{j[0]}_lat{j[1]}_refined_cm").exists()]
    running: dict[int, tuple[tuple[str, int], subprocess.Popen]] = {}
    failed = 0
    while pending or running:
        for gpu in list(running):
            job, proc = running[gpu]
            code = proc.poll()
            if code is None:
                continue
            log(f"{'DONE' if code == 0 else 'FAIL'} {job[0]}_lat{job[1]} gpu={gpu} exit={code}")
            failed += int(code != 0)
            del running[gpu]
        while pending and len(running) < args.max_concurrent:
            gpu = pick_gpu(set(running))
            if gpu is None:
                break
            job = pending.pop(0)
            log(f"START {job[0]}_lat{job[1]} gpu={gpu} remaining={len(pending)}")
            running[gpu] = (job, run_chain(job[0], job[1], gpu))
            time.sleep(5)
        time.sleep(30)
    log(f"chains finished failed={failed}")

    log("STAGE report")
    subprocess.run([PY, str(ROOT / "scripts/ladder_public_si_report.py"), "--out", str(CH / "ladder_public_si_report.json")], check=False)
    log("DONE public-337 SI pipeline")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
