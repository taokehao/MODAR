#!/usr/bin/env python3
"""Train the 13–20 specialist on leftover GPUs of the current node.

Picks GPUs with util<=15% and >=12 GiB free. Does not kill VASP/ColabFold.
Survives the launching shell via setsid. Paper mix run_dir is never written.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
CONDA_PY = Path("/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python")
DATA_DIR = ROOT / "data_prep/outputs/crystal_language_unified_tierA_v2"
BASE_RUN = ROOT / "crystal_language/runs/tierA_s2_di"
RUN_DIR = ROOT / "crystal_language/runs/tierA_s2_di_n13_20"
LOG_DIR = ROOT / "logs"
PIDFILE = LOG_DIR / "n13_20_train.pid"
LOCK = Path("/tmp/tkh_gpu_pick.lock")

MIN_FREE_MIB = 12000
MAX_UTIL_PCT = 15
TARGET_BATCH = 192
MICRO_BATCH = 12
MAX_STEPS = 18000
LR = "1e-4"

COMMON = [
    str(CONDA_PY),
    "-u",
    "crystal_language/train.py",
    "--arch",
    "v2",
    "--rep",
    "v2",
    "--data-dir",
    str(DATA_DIR),
    "--out-dir",
    str(RUN_DIR),
    "--condition-set",
    "pxrd_formula",
    "--use-lattice",
    "always",
    "--target-mode",
    "full",
    "--block-size",
    "1024",
    "--n-layer",
    "16",
    "--n-head",
    "16",
    "--n-embd",
    "1024",
    "--spectra-encoder",
    "di",
    "--spectra-max-peaks",
    "128",
    "--spectra-encoder-layers",
    "3",
    "--batch-size",
    str(MICRO_BATCH),
    "--lr",
    LR,
    "--max-steps",
    str(MAX_STEPS),
    "--warmup-steps",
    "500",
    "--eval-every",
    "1000",
    "--save-every",
    "2000",
    "--num-workers",
    "8",
    "--amp",
    "bf16",
    "--cond-adanorm",
    "--cond-drop",
    "0.1",
    "--coord-value-emb",
    "--noise-profile",
    "experimental",
    "--lazy-data",
    "auto",
    "--init-from",
    str(BASE_RUN / "checkpoint_last.pt"),
    "--min-sites",
    "13",
    "--max-sites",
    "20",
]


def gpu_status() -> list[tuple[int, int, int]]:
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
    out = []
    for line in raw.strip().splitlines():
        idx, free, util = [p.strip() for p in line.split(",")]
        out.append((int(idx), int(free), int(util)))
    return out


def gpu_compute_apps() -> dict[int, list[tuple[int, str]]]:
    """gpu_index -> [(pid, process_name), ...] for resident compute processes."""
    smi_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    }
    mapping = {}
    raw_gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True,
        env=smi_env,
    )
    bus_to_idx = {}
    for line in raw_gpu.strip().splitlines():
        idx, bus = [p.strip() for p in line.split(",", 1)]
        bus_to_idx[bus.lower()] = int(idx)
        mapping[int(idx)] = []
    raw_apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
        env=smi_env,
    )
    for line in raw_apps.strip().splitlines():
        if not line.strip():
            continue
        bus, pid_s, name = [p.strip() for p in line.split(",", 2)]
        idx = bus_to_idx.get(bus.lower())
        if idx is None:
            continue
        mapping[idx].append((int(pid_s), name))
    return mapping


def leftover_gpus() -> list[int]:
    """Empty leftover cards only.

    Long training must not share a GPU with resident VASP/ColabFold even when
    those jobs are briefly idle (util<=15%). Eval leftover-sharing is fine;
    a 10h specialist run is not.
    """
    LOCK.touch(exist_ok=True)
    with LOCK.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            apps = gpu_compute_apps()
            picked = []
            for idx, free, util in gpu_status():
                if util > MAX_UTIL_PCT or free < MIN_FREE_MIB:
                    continue
                if apps.get(idx):
                    continue
                picked.append(idx)
            return picked
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def claim_pidfile() -> object:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = PIDFILE.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        other = handle.read().strip() or "?"
        print(f"n13-20 trainer already running pid={other}", flush=True)
        sys.exit(0)
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def accum_for(ngpu: int) -> int:
    # Keep effective batch close to 192: micro * accum * ngpu.
    return max(1, round(TARGET_BATCH / (MICRO_BATCH * ngpu)))


def run(cmd: list[str], env: dict[str, str], log) -> int:
    log.write(" ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return proc.wait()


def main() -> int:
    if not (BASE_RUN / "checkpoint_last.pt").exists():
        print(f"FATAL: missing init {BASE_RUN / 'checkpoint_last.pt'}", flush=True)
        return 2
    mix_mtime = (BASE_RUN / "checkpoint_last.pt").stat().st_mtime
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tok_src = BASE_RUN / "tokenizer.json"
    tok_dst = RUN_DIR / "tokenizer.json"
    if not tok_dst.exists():
        tok_dst.write_bytes(tok_src.read_bytes())
        print(f"copied tokenizer.json into {RUN_DIR}", flush=True)

    pid_handle = claim_pidfile()
    log_path = LOG_DIR / f"n13_20_train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "8"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("NVIDIA_VISIBLE_DEVICES", None)

    with log_path.open("w") as log:
        log.write(f"n13-20 leftover train {time.strftime('%F %T')}\n")
        log.flush()
        print(f"log={log_path}", flush=True)
        print("=== prepare-only (CPU index cache) ===", flush=True)
        code = run(
            COMMON + ["--prepare-only", "--device", "cpu", "--grad-accum-steps", "4"],
            env,
            log,
        )
        if code != 0:
            print(f"prepare-only failed exit={code}", flush=True)
            return code

        gpus = leftover_gpus()
        if not gpus:
            print("no leftover GPU (util<=15% and >=12 GiB free); waiting", flush=True)
            while not gpus:
                time.sleep(30)
                gpus = leftover_gpus()
        ngpu = len(gpus)
        accum = accum_for(ngpu)
        gpu_csv = ",".join(str(g) for g in gpus)
        env["CUDA_VISIBLE_DEVICES"] = gpu_csv
        print(
            f"train gpus={gpu_csv} micro={MICRO_BATCH} accum={accum} "
            f"eff={MICRO_BATCH * accum * ngpu} steps={MAX_STEPS}",
            flush=True,
        )
        log.write(
            f"train gpus={gpu_csv} micro={MICRO_BATCH} accum={accum} "
            f"eff={MICRO_BATCH * accum * ngpu}\n"
        )
        log.flush()
        train_cmd = COMMON + ["--grad-accum-steps", str(accum)]
        if ngpu == 1:
            cmd = train_cmd + ["--device", "cuda:0"]
        else:
            cmd = [
                str(CONDA_PY),
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={ngpu}",
            ] + train_cmd[2:]
        code = run(cmd, env, log)
        mix_mtime_after = (BASE_RUN / "checkpoint_last.pt").stat().st_mtime
        if mix_mtime_after != mix_mtime:
            print(
                f"WARNING: paper mix checkpoint mtime changed {mix_mtime} -> {mix_mtime_after}",
                flush=True,
            )
        print(f"train exit={code}", flush=True)
        try:
            pid_handle.close()
            PIDFILE.unlink(missing_ok=True)
        except OSError:
            pass
        return code


if __name__ == "__main__":
    sys.exit(main())
