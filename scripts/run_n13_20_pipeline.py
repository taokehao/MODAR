#!/usr/bin/env python3
"""Hang until the 13–20 specialist finishes, then generate / score / merge / report.

Prints training progress while waiting. After checkpoint_last.pt:
  1. SI generation on CNRS 21 + RRUFF 117 (T=1.0 / k=50 / N=100)
  2. CHGNet + MACE on those candidates (leftover GPU)
  3. Route: mix model for n<13, specialist for 13–20
  4. Print vs RealPXRD-Solver 101/118 and 199/245

Does not overwrite the paper mix run. Does not kill VASP/ColabFold.
"""
from __future__ import annotations

import fcntl
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
PY = Path("/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python")
LOG_DIR = ROOT / "logs"
RUN_DIR = ROOT / "crystal_language/runs/tierA_s2_di_n13_20"
CKPT = RUN_DIR / "checkpoint_last.pt"
MIX_CKPT = ROOT / "crystal_language/runs/tierA_s2_di/checkpoint_last.pt"
EVAL = ROOT / "crystal_language/eval_results"
CHG = EVAL / "chgnet"
PIDFILE = LOG_DIR / "n13_20_pipeline.pid"
LOCK = Path("/tmp/tkh_gpu_pick.lock")

MAX_STEPS = 18000
MIN_FREE_MIB = 12000
MAX_UTIL_PCT = 15
POLL_S = 60

CNRS_SPEC = EVAL / "rps_cnrs126_si_n1320"
RRUFF_SPEC = EVAL / "rps_rruff269_si_n1320"
CNRS_MIX = EVAL / "rps_cnrs126_si_profile_sweep_t10k50"
RRUFF_MIX = EVAL / "rps_rruff269_si_peaks_sweep_t10k50"
CNRS_ROUTED = EVAL / "rps_cnrs126_si_routed_n1320"
RRUFF_ROUTED = EVAL / "rps_rruff269_si_routed_n1320"

STEP_RE = re.compile(
    r"^step (?P<step>\d+) loss (?P<loss>[-0-9.]+) lr (?P<lr>[-0-9.e+]+) elapsed (?P<elapsed>[-0-9.]+)s"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    log("RUN " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def train_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "crystal_language/train.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any("--max-sites 20" in line and "--min-sites 13" in line for line in out.splitlines())


def latest_train_log() -> Path | None:
    logs = sorted(LOG_DIR.glob("n13_20_train_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def last_step(path: Path | None) -> tuple[int, str] | None:
    if path is None or not path.exists():
        return None
    last = None
    with path.open(errors="replace") as handle:
        for line in handle:
            m = STEP_RE.match(line.strip())
            if m:
                last = m
    if last is None:
        return None
    step = int(last.group("step"))
    elapsed = float(last.group("elapsed"))
    rate = elapsed / max(step, 1)
    eta_h = (MAX_STEPS - step) * rate / 3600.0
    text = (
        f"step {step}/{MAX_STEPS} loss {last.group('loss')} "
        f"elapsed {elapsed:.0f}s eta {eta_h:.1f}h"
    )
    return step, text


def gpu_line() -> str:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        env={
            k: v
            for k, v in os.environ.items()
            if k not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        },
    )
    return " ".join(f"{a.strip()}/{b.strip()}MiB/{c.strip()}%" for a, b, c in
                    (ln.split(",") for ln in raw.strip().splitlines()))


def leftover_empty() -> list[int]:
    smi_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    }
    gpus = []
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        env=smi_env,
    ).strip().splitlines():
        idx, free, util = [p.strip() for p in line.split(",")]
        gpus.append((int(idx), int(free), int(util)))
    bus_to_idx = {}
    occupied: dict[int, int] = {i: 0 for i, _, _ in gpus}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True,
        env=smi_env,
    ).strip().splitlines():
        idx, bus = [p.strip() for p in line.split(",", 1)]
        bus_to_idx[bus.lower()] = int(idx)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"],
        text=True,
        env=smi_env,
    )
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        bus, _pid = [p.strip() for p in line.split(",", 1)]
        idx = bus_to_idx.get(bus.lower())
        if idx is not None:
            occupied[idx] = occupied.get(idx, 0) + 1
    picked = []
    for idx, free, util in gpus:
        if util <= MAX_UTIL_PCT and free >= MIN_FREE_MIB and occupied.get(idx, 0) == 0:
            picked.append(idx)
    return picked


def wait_leftover_gpu() -> int:
    log("waiting for an empty leftover GPU (util<=15%, >=12 GiB, no resident job)")
    while True:
        LOCK.touch(exist_ok=True)
        with LOCK.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                gpus = leftover_empty()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        if gpus:
            log(f"leftover empty gpus={gpus}")
            return gpus[0]
        time.sleep(30)


def claim_pidfile() -> object:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = PIDFILE.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        other = handle.read().strip() or "?"
        print(f"pipeline already running pid={other}", flush=True)
        sys.exit(0)
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def wait_train() -> int:
    log("STAGE wait-train")
    mix_mtime = MIX_CKPT.stat().st_mtime if MIX_CKPT.exists() else None
    started = time.time()
    while True:
        alive = train_running()
        info = last_step(latest_train_log())
        status = info[1] if info else "no step line yet"
        log(f"train_alive={alive} {status}  gpus {gpu_line()}")
        if CKPT.exists() and not alive:
            log(f"STAGE train-done  ckpt={CKPT} mtime={time.ctime(CKPT.stat().st_mtime)}")
            if mix_mtime is not None and MIX_CKPT.stat().st_mtime != mix_mtime:
                log("WARNING paper mix checkpoint mtime changed")
            return 0
        if (
            time.time() - started > 180
            and not alive
            and not CKPT.exists()
            and not (LOG_DIR / "n13_20_train.pid").exists()
        ):
            log("FAIL trainer gone without checkpoint_last.pt")
            return 1
        time.sleep(POLL_S)


def generate() -> int:
    log("STAGE generate CNRS-21 + RRUFF-117  T=1.0 k=50 N=100")
    return run(
        [
            str(PY),
            "-u",
            str(ROOT / "scripts/run_rps_si_di_local.py"),
            "--only",
            "cnrs_n1320",
            "--only",
            "rruff_n1320",
            "--pidfile",
            str(LOG_DIR / "n13_20_eval.pid"),
        ]
    )


def score(eval_dir: Path, out: Path, backend: str, gpu: int) -> None:
    if out.exists() and out.stat().st_size > 0:
        log(f"SKIP score exists {out}")
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    code = run(
        [
            str(PY),
            "-u",
            str(ROOT / "scripts/score_candidates_mlip.py"),
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--backend",
            backend,
            "--resume",
        ],
        env=env,
    )
    if code != 0:
        raise RuntimeError(f"score {backend} {eval_dir} exit={code}")


def merge_one(
    mix_eval: Path,
    spec_eval: Path,
    out_eval_dir: Path,
    mix_energy: Path,
    spec_energy: Path,
    out_energy: Path,
) -> None:
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    out_eval = out_eval_dir / "test_pxrd_formula_lat1_full_inject0_cd1_rr1_start0_n_s100.jsonl"
    code = run(
        [
            str(PY),
            "-u",
            str(ROOT / "scripts/merge_rps_routed_eval.py"),
            "--mix-eval",
            str(mix_eval),
            "--spec-eval",
            str(spec_eval),
            "--out-eval",
            str(out_eval),
            "--mix-energy",
            str(mix_energy),
            "--spec-energy",
            str(spec_energy),
            "--out-energy",
            str(out_energy),
        ]
    )
    if code != 0:
        raise RuntimeError(f"merge {out_eval_dir} exit={code}")


def analyze(name: str, eval_dir: Path, energy: Path, rps: str) -> None:
    out = CHG / f"{eval_dir.name}_{energy.stem}_summary.json"
    run(
        [
            str(PY),
            "-u",
            str(ROOT / "scripts/analyze_rps_step1.py"),
            "--name",
            name,
            "--base-eval",
            str(eval_dir),
            "--base-energy",
            str(energy),
            "--rps",
            rps,
            "--out",
            str(out),
        ]
    )


def main() -> int:
    os.chdir(ROOT)
    CHG.mkdir(parents=True, exist_ok=True)
    handle = claim_pidfile()
    log("n13-20 pipeline start  paper mix frozen")
    try:
        code = wait_train()
        if code != 0:
            return code
        code = generate()
        if code != 0:
            log(f"FAIL generate exit={code}")
            return code
        log("STAGE score CHGNet+MACE on specialist candidates")
        gpu = wait_leftover_gpu()
        jobs = [
            (CNRS_SPEC, CHG / "rps_cnrs126_si_n1320_chgnet.jsonl", "chgnet"),
            (CNRS_SPEC, CHG / "rps_cnrs126_si_n1320_mace.jsonl", "mace"),
            (RRUFF_SPEC, CHG / "rps_rruff269_si_n1320_chgnet.jsonl", "chgnet"),
            (RRUFF_SPEC, CHG / "rps_rruff269_si_n1320_mace.jsonl", "mace"),
        ]
        for eval_dir, out, backend in jobs:
            score(eval_dir, out, backend, gpu)

        log("STAGE merge routed pools (mix n<13 + specialist 13-20)")
        merge_one(
            CNRS_MIX,
            CNRS_SPEC,
            CNRS_ROUTED,
            CHG / "rps_cnrs126_si_profile_sweep_t10k50.jsonl",
            CHG / "rps_cnrs126_si_n1320_chgnet.jsonl",
            CHG / "rps_cnrs126_si_routed_n1320_chgnet.jsonl",
        )
        merge_one(
            CNRS_MIX,
            CNRS_SPEC,
            CNRS_ROUTED,
            CHG / "rps_cnrs126_si_profile_sweep_t10k50_mace.jsonl",
            CHG / "rps_cnrs126_si_n1320_mace.jsonl",
            CHG / "rps_cnrs126_si_routed_n1320_mace.jsonl",
        )
        merge_one(
            RRUFF_MIX,
            RRUFF_SPEC,
            RRUFF_ROUTED,
            CHG / "rps_rruff269_si_peaks_sweep_t10k50.jsonl",
            CHG / "rps_rruff269_si_n1320_chgnet.jsonl",
            CHG / "rps_rruff269_si_routed_n1320_chgnet.jsonl",
        )
        merge_one(
            RRUFF_MIX,
            RRUFF_SPEC,
            RRUFF_ROUTED,
            CHG / "rps_rruff269_si_peaks_sweep_t10k50_mace.jsonl",
            CHG / "rps_rruff269_si_n1320_mace.jsonl",
            CHG / "rps_rruff269_si_routed_n1320_mace.jsonl",
        )

        log("STAGE analyze vs RPS")
        analyze("CNRS specialist-only n13-20 CHGNet fusion", CNRS_SPEC, CHG / "rps_cnrs126_si_n1320_chgnet.jsonl", "cnrs")
        analyze("CNRS specialist-only n13-20 MACE energy", CNRS_SPEC, CHG / "rps_cnrs126_si_n1320_mace.jsonl", "cnrs")
        analyze("CNRS routed mix+n1320 CHGNet fusion", CNRS_ROUTED, CHG / "rps_cnrs126_si_routed_n1320_chgnet.jsonl", "cnrs")
        analyze("CNRS routed mix+n1320 MACE energy/fusion", CNRS_ROUTED, CHG / "rps_cnrs126_si_routed_n1320_mace.jsonl", "cnrs")
        analyze("RRUFF specialist-only n13-20 CHGNet fusion", RRUFF_SPEC, CHG / "rps_rruff269_si_n1320_chgnet.jsonl", "rruff")
        analyze("RRUFF specialist-only n13-20 MACE energy", RRUFF_SPEC, CHG / "rps_rruff269_si_n1320_mace.jsonl", "rruff")
        analyze("RRUFF routed mix+n1320 CHGNet fusion", RRUFF_ROUTED, CHG / "rps_rruff269_si_routed_n1320_chgnet.jsonl", "rruff")
        analyze("RRUFF routed mix+n1320 MACE energy/fusion", RRUFF_ROUTED, CHG / "rps_rruff269_si_routed_n1320_mace.jsonl", "rruff")
        log("DONE n13-20 pipeline")
        return 0
    finally:
        try:
            handle.close()
            PIDFILE.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
