#!/usr/bin/env python3
"""When the 13–20 specialist finishes, run routed SI generation on leftover GPUs.

Waits for checkpoint_last.pt and the trainer to exit 0, then launches
run_rps_si_di_local.py for cnrs_n1320 + rruff_n1320 (T=1.0 / k=50 / N=100).
Does not score MLIP; that stays a separate leftover job after jsonl exists.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
CONDA_PY = Path("/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python")
RUN_DIR = ROOT / "crystal_language/runs/tierA_s2_di_n13_20"
CKPT = RUN_DIR / "checkpoint_last.pt"
LOG_DIR = ROOT / "logs"
DISPATCHER = ROOT / "scripts/run_rps_si_di_local.py"


def train_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "crystal_language/train.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    for line in out.splitlines():
        if "--max-sites 20" in line and "--min-sites 13" in line:
            return True
    return False


def main() -> int:
    pidfile = LOG_DIR / "n13_20_train.pid"
    print(f"watching {CKPT}", flush=True)
    started = time.time()
    while True:
        alive = train_running()
        if CKPT.exists() and not alive:
            break
        if (
            time.time() - started > 120
            and not CKPT.exists()
            and not alive
            and not pidfile.exists()
        ):
            print("trainer exited without checkpoint_last.pt", flush=True)
            return 1
        time.sleep(60)
    print(f"specialist ready {CKPT} mtime={time.ctime(CKPT.stat().st_mtime)}", flush=True)
    cmd = [
        str(CONDA_PY),
        "-u",
        str(DISPATCHER),
        "--only",
        "cnrs_n1320",
        "--only",
        "rruff_n1320",
        "--pidfile",
        str(LOG_DIR / "n13_20_eval.pid"),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
