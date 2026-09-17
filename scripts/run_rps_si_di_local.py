#!/usr/bin/env python3
"""Run SI d–I shards on leftover GPUs of the current node, skipping Slurm.

Picks GPUs with low utilization and enough free memory. Does not kill other
users' processes. Skips a shard if its summary exists or another eval already
holds that start-index. Survives the launching shell via setsid.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/tkh/python-project/multimodal_csp-v2")
CONDA_PY = Path("/home/tkh/software/miniconda3/envs/pxrd_crystal_lm/bin/python")
RUN_DIR = ROOT / "crystal_language/runs/tierA_s2_di"
LOG_DIR = ROOT / "logs"
LOCK = Path("/tmp/tkh_gpu_pick.lock")
PIDFILE = LOG_DIR / "rps_si_local_dispatcher.pid"

DI_RUN = ROOT / "crystal_language/runs/tierA_s2_di"
RPS_LAT_RUN = ROOT / "crystal_language/runs/tierA_s2_di_rps_formula_lat"
N13_20_RUN = ROOT / "crystal_language/runs/tierA_s2_di_n13_20"
NOISY_RUN = ROOT / "crystal_language/runs/tierA_s2_noisy_exp"
REALP_6M_RUN = ROOT / "crystal_language/runs/realpxrd_v2_s1_ada_cve"
CNRS_PROFILE = ROOT / "data_prep/outputs/rps_cnrs126_si_profile/crystal_language_v2"

SETS = {
    "cnrs_profile": {
        "data": CNRS_PROFILE,
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_profile_di",
        "n": 126,
        "run_dir": DI_RUN,
    },
    "cnrs_profile_noisy": {
        "data": CNRS_PROFILE,
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_profile_noisy",
        "n": 126,
        "run_dir": NOISY_RUN,
    },
    "cnrs_profile_6m": {
        "data": CNRS_PROFILE,
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_profile_6m",
        "n": 126,
        "run_dir": REALP_6M_RUN,
    },
    "cnrs_profile_nxx": {
        "data": CNRS_PROFILE,
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_profile_di_nxx",
        "n": 126,
        "run_dir": DI_RUN,
        "condition_set": "pxrd_neutron_xrf",
        "constrain": "none",
    },
    "rruff_peaks_nxx": {
        "data": ROOT / "data_prep/outputs/rps_rruff269_si_peaks/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_peaks_di_nxx",
        "n": 269,
        "run_dir": DI_RUN,
        "condition_set": "pxrd_neutron_xrf",
        "constrain": "none",
    },
    "rruff_peaks": {
        "data": ROOT / "data_prep/outputs/rps_rruff269_si_peaks/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_peaks_di",
        "n": 269,
        "run_dir": DI_RUN,
    },
    "rruff_profile": {
        "data": ROOT / "data_prep/outputs/rps_rruff269_si_profile/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_profile_di",
        "n": 269,
        "run_dir": DI_RUN,
    },
    "cnrs_covermiss": {
        "data": ROOT / "data_prep/outputs/rps_covermiss_si/cnrs_profile/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_covermiss_di",
        "n": 9,
        "run_dir": DI_RUN,
    },
    "rruff_covermiss": {
        "data": ROOT / "data_prep/outputs/rps_covermiss_si/rruff_peaks/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_covermiss_di",
        "n": 41,
        "run_dir": DI_RUN,
    },
    # Step-1 adaptive N: extra 400 samples at the honest T=1.0 / k=50 setting,
    # independent seed so they do not replay the original 100. Merge offline.
    "cnrs_adapt_n400": {
        "data": ROOT / "data_prep/outputs/rps_adapt_n/cnrs_profile_n13/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_adapt_n400",
        "n": 21,
        "run_dir": DI_RUN,
        "temperature": "1.0",
        "top_k_sampling": "50",
        "seed": 43,
    },
    "rruff_adapt_n400": {
        "data": ROOT / "data_prep/outputs/rps_adapt_n/rruff_peaks_n13/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_adapt_n400",
        "n": 117,
        "run_dir": DI_RUN,
        "temperature": "1.0",
        "top_k_sampling": "50",
        "seed": 43,
    },
    # Paper mix model is unchanged. This specialist was trained only on
    # PXRD+formula+known lattice and is used solely for the RPS comparison.
    "cnrs_rps_lat": {
        "data": CNRS_PROFILE,
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_profile_rpslat",
        "n": 126,
        "run_dir": RPS_LAT_RUN,
        "temperature": "1.0",
        "top_k_sampling": "50",
        "seed": 42,
    },
    "rruff_rps_lat": {
        "data": ROOT / "data_prep/outputs/rps_rruff269_si_peaks/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_peaks_rpslat",
        "n": 269,
        "run_dir": RPS_LAT_RUN,
        "temperature": "1.0",
        "top_k_sampling": "50",
        "seed": 42,
    },
    # 13–20 specialist on the SI n>=13 slice (all SI cells are n<=20).
    # Merge with mix T=1.0 / k=50 rows for n<13 via merge_rps_routed_eval.py.
    "cnrs_n1320": {
        "data": ROOT / "data_prep/outputs/rps_adapt_n/cnrs_profile_n13/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_cnrs126_si_n1320",
        "n": 21,
        "run_dir": N13_20_RUN,
        "temperature": "1.0",
        "top_k_sampling": "50",
        "seed": 42,
    },
    "rruff_n1320": {
        "data": ROOT / "data_prep/outputs/rps_adapt_n/rruff_peaks_n13/crystal_language_v2",
        "out": ROOT / "crystal_language/eval_results/rps_rruff269_si_n1320",
        "n": 117,
        "run_dir": N13_20_RUN,
        "temperature": "1.0",
        "top_k_sampling": "50",
        "seed": 42,
    },
}

# Decoding sweep: the coverage curve saturates well below their 20-sample
# coverage, and s@1 is 57% against their 80%, so the sampling temperature that
# was tuned on the synthetic ladder may simply be wrong for these sets.
RRUFF_PEAKS = ROOT / "data_prep/outputs/rps_rruff269_si_peaks/crystal_language_v2"
# Independent of the RPS releases, matched to their n_sites band, used only to
# pick the sampling temperature so the CNRS/RRUFF numbers stay held out.
TEMPSELECT = {
    "public_nle20": (ROOT / "data_prep/outputs/rps_tempselect/public_nle20/crystal_language_v2", 69),
    "tierA_nle20": (ROOT / "data_prep/outputs/rps_tempselect/tierA_nle20/crystal_language_v2", 192),
}

for _t, _k in ((0.2, 20), (0.4, 20), (0.6, 20), (0.8, 20), (1.0, 50), (1.2, 100), (1.4, 200)):
    _tag = f"t{str(_t).replace('.', '')}k{_k}"
    SETS[f"cnrs_sweep_{_tag}"] = {
        "data": CNRS_PROFILE,
        "out": ROOT / f"crystal_language/eval_results/rps_cnrs126_si_profile_sweep_{_tag}",
        "n": 126,
        "run_dir": DI_RUN,
        "temperature": str(_t),
        "top_k_sampling": str(_k),
    }
    # RRUFF T20 is 19 short and purely coverage-bound, so the diversity end of
    # the sweep is what matters there.
    SETS[f"rruff_sweep_{_tag}"] = {
        "data": RRUFF_PEAKS,
        "out": ROOT / f"crystal_language/eval_results/rps_rruff269_si_peaks_sweep_{_tag}",
        "n": 269,
        "run_dir": DI_RUN,
        "temperature": str(_t),
        "top_k_sampling": str(_k),
    }
    for _sel, (_dir, _n) in TEMPSELECT.items():
        SETS[f"sel_{_sel}_{_tag}"] = {
            "data": _dir,
            "out": ROOT / f"crystal_language/eval_results/tempselect_{_sel}_{_tag}",
            "n": _n,
            "run_dir": DI_RUN,
            "temperature": str(_t),
            "top_k_sampling": str(_k),
        }

# Public experimental set (337 targets) re-encoded under the SI protocol
# (build_public_exp_pxrd_si.py): full condition ladder, lattice-free and
# known-lattice, same decoding as the RPS benchmark runs. One dir per rung.
PUB337_SI = ROOT / "data_prep/outputs/public_exp_pxrd_si/crystal_language_v2"
LADDER_CONDS = (
    "pxrd",
    "pxrd_xrf",
    "pxrd_neutron",
    "pxrd_neutron_xrf",
    "pxrd_neutron_xrf_xps",
    "pxrd_formula",
)
for _cond in LADDER_CONDS:
    for _lat in (1, 0):
        SETS[f"pub337si_{_cond}_lat{_lat}"] = {
            "data": PUB337_SI,
            "out": ROOT / f"crystal_language/eval_results/ladder_public_si/{_cond}_lat{_lat}",
            "n": 337,
            "run_dir": DI_RUN,
            "condition_set": _cond,
            "constrain": "formula" if _cond == "pxrd_formula" else "none",
            "use_lattice": bool(_lat),
            "temperature": "1.0",
            "top_k_sampling": "50",
        }

SHARD_SIZE = 32
NUM_SAMPLES = 100
SAMPLES_PER_BATCH = 10
TEMPERATURE = "0.8"
TOP_K_SAMPLING = "20"
SEED = 42
MIN_FREE_MIB = 12000
MAX_UTIL_PCT = 15
EXCLUDE_GPUS: set[int] = set()


def gpu_status() -> list[tuple[int, int, int]]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        env={k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
    )
    out = []
    for line in raw.strip().splitlines():
        idx, free, util = [p.strip() for p in line.split(",")]
        out.append((int(idx), int(free), int(util)))
    return out


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return {}
    out = {}
    for item in raw:
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        out[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return out


def out_to_set(out_dir: str) -> str | None:
    """Match a running eval to a preset by output directory, not data-dir.

    Several presets share the CNRS SI profile jsonl; keying on data-dir would
    make a noisy-model shard look like a 6M shard and skip it.
    """
    out = str(Path(out_dir).resolve())
    for name, spec in SETS.items():
        if str(spec["out"].resolve()) == out:
            return name
    return None


def scan_running_evals() -> dict[tuple[str, int], int]:
    """(set_name, start) -> gpu for evaluate_generation parents we launched."""
    found: dict[tuple[str, int], int] = {}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = _cmdline(pid)
        if "crystal_language.evaluate_" not in cmd and "evaluate_generation.py" not in cmd:
            continue
        if "--start-index" not in cmd:
            continue
        parts = cmd.split()
        start = None
        out_dir = None
        for i, tok in enumerate(parts):
            if tok == "--start-index" and i + 1 < len(parts):
                start = int(parts[i + 1])
            elif tok == "--out-dir" and i + 1 < len(parts):
                out_dir = parts[i + 1]
        if start is None or out_dir is None:
            continue
        name = out_to_set(out_dir)
        if name is None:
            continue
        env = _environ(pid)
        gpu_s = env.get("CUDA_VISIBLE_DEVICES", "")
        if gpu_s == "" or "," in gpu_s:
            continue
        # Skip ProcessPoolExecutor children: they share the parent cmdline.
        try:
            ppid = 0
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
        except (OSError, IndexError, ValueError):
            ppid = 0
        parent_cmd = _cmdline(ppid)
        if "crystal_language.evaluate_" in parent_cmd or "evaluate_generation.py" in parent_cmd:
            continue
        found[(name, start)] = int(gpu_s)
    return found


def pick_gpu(busy: set[int]) -> int | None:
    LOCK.touch(exist_ok=True)
    with LOCK.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            ranked = []
            for idx, free, util in gpu_status():
                if idx in busy or idx in EXCLUDE_GPUS:
                    continue
                if util > MAX_UTIL_PCT or free < MIN_FREE_MIB:
                    continue
                ranked.append((free, -util, idx))
            if not ranked:
                return None
            ranked.sort(reverse=True)
            return ranked[0][2]
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stem(name: str, start: int) -> str:
    spec = SETS[name]
    cond = spec.get("condition_set", "pxrd_formula")
    cd = int(spec.get("constrain", "formula") != "none")
    lat = int(spec.get("use_lattice", True))
    return (
        f"test_{cond}_lat{lat}_full_inject0_cd{cd}_rr1"
        f"_start{start}_n{SHARD_SIZE}_s{NUM_SAMPLES}"
    )


def shards() -> list[tuple[str, int]]:
    jobs = []
    for name, spec in SETS.items():
        for start in range(0, spec["n"], SHARD_SIZE):
            jobs.append((name, start))
    return jobs


def run_shard(name: str, start: int, gpu: int) -> subprocess.Popen:
    spec = SETS[name]
    spec["out"].mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"rps_si_local_{name}_s{start}_gpu{gpu}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "4"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # run as a module: the eval driver may exist only as a sourceless .pyc
    cmd = [
        str(CONDA_PY),
        "-u",
        "-m",
        f"crystal_language.{spec.get('module', 'evaluate_generation')}",
        "--run-dir",
        str(spec.get("run_dir", RUN_DIR)),
        "--checkpoint",
        "checkpoint_last.pt",
        "--data-dir",
        str(spec["data"]),
        "--out-dir",
        str(spec["out"]),
        "--split",
        "test",
        "--condition-set",
        spec.get("condition_set", "pxrd_formula"),
        *(["--use-lattice"] if spec.get("use_lattice", True) else []),
        "--target-mode",
        "full",
        "--constrain",
        spec.get("constrain", "formula"),
        "--rerank",
        "--rerank-workers",
        "2",
        "--amp",
        "bf16",
        "--noise-profile",
        "none",
        "--noise-seed",
        "20260706",
        "--store-structures",
        "--start-index",
        str(start),
        "--max-records",
        str(SHARD_SIZE),
        "--num-samples",
        str(NUM_SAMPLES),
        "--top-ks",
        "1,10,20,50",
        "--samples-per-batch",
        str(SAMPLES_PER_BATCH),
        "--temperature",
        str(spec.get("temperature", TEMPERATURE)),
        "--top-k-sampling",
        str(spec.get("top_k_sampling", TOP_K_SAMPLING)),
        "--max-new-tokens",
        "1024",
        "--seed",
        str(spec.get("seed", SEED)),
    ]
    handle = log.open("w")
    handle.write(f"gpu={gpu} set={name} start={start} {time.strftime('%F %T')}\n")
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def claim_pidfile() -> object:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = PIDFILE.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        other = handle.read().strip() or "?"
        print(f"dispatcher already running pid={other}", flush=True)
        sys.exit(0)
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    global SETS, PIDFILE, NUM_SAMPLES, SHARD_SIZE, TEMPERATURE, TOP_K_SAMPLING, SEED, EXCLUDE_GPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--pidfile", default="")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--temperature", default=TEMPERATURE)
    ap.add_argument("--top-k-sampling", default=TOP_K_SAMPLING)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude-gpus", default="")
    ap.add_argument("--only-gpus", default="")
    args = ap.parse_args()
    NUM_SAMPLES = args.num_samples
    SHARD_SIZE = args.shard_size
    TEMPERATURE = str(args.temperature)
    TOP_K_SAMPLING = str(args.top_k_sampling)
    SEED = int(args.seed)
    if args.exclude_gpus:
        EXCLUDE_GPUS = {int(x) for x in args.exclude_gpus.split(",") if x.strip()}
    if args.only_gpus:
        allow = {int(x) for x in args.only_gpus.split(",") if x.strip()}
        EXCLUDE_GPUS |= {i for i in range(8) if i not in allow}
    if args.only:
        missing = [n for n in args.only if n not in SETS]
        if missing:
            print(f"unknown sets {missing}", flush=True)
            return 2
        SETS = {n: SETS[n] for n in args.only}
    if args.pidfile:
        PIDFILE = Path(args.pidfile)

    pid_handle = claim_pidfile()
    existing = scan_running_evals()
    pending: list[tuple[str, int]] = []
    for name, start in shards():
        summary = SETS[name]["out"] / f"{stem(name, start)}_summary.json"
        jsonl = SETS[name]["out"] / f"{stem(name, start)}.jsonl"
        if summary.exists() and jsonl.exists() and jsonl.stat().st_size > 0:
            print(f"SKIP {name} start={start} (summary)", flush=True)
            continue
        if summary.exists() and not jsonl.exists():
            print(f"STALE {name} start={start} (summary without jsonl), rerun", flush=True)
            summary.unlink()
        if (name, start) in existing:
            print(f"SKIP {name} start={start} (already running gpu={existing[(name, start)]})", flush=True)
            continue
        pending.append((name, start))
    print(f"{len(pending)} shards to run  already_running={len(existing)}", flush=True)
    running: dict[int, tuple[str, int, subprocess.Popen]] = {}
    failed = 0
    while pending or running:
        existing = scan_running_evals()
        busy = set(running) | set(existing.values())
        for gpu in list(running):
            name, start, proc = running[gpu]
            code = proc.poll()
            if code is None:
                continue
            tag = f"{name} start={start} gpu={gpu}"
            if code == 0:
                print(f"DONE {tag}", flush=True)
            else:
                print(f"FAIL {tag} exit={code}", flush=True)
                failed += 1
            del running[gpu]
        while pending:
            gpu = pick_gpu(busy)
            if gpu is None:
                break
            name, start = pending.pop(0)
            if (name, start) in scan_running_evals():
                print(f"SKIP {name} start={start} (raced)", flush=True)
                continue
            print(f"START {name} start={start} gpu={gpu}  remaining={len(pending)}", flush=True)
            running[gpu] = (name, start, run_shard(name, start, gpu))
            busy.add(gpu)
            time.sleep(3)
        if pending or running:
            time.sleep(20)
    print(f"all shards finished  failed={failed}", flush=True)
    try:
        pid_handle.close()
        PIDFILE.unlink(missing_ok=True)
    except OSError:
        pass
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
