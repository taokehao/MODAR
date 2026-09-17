#!/usr/bin/env python
"""Top-k generation evaluation for the crystal language model.

For each input record, this script samples multiple crystal token sequences,
parses them, optionally converts them to ``pymatgen.Structure``, and computes
Top-1 / Top-20 / Top-50 style success metrics. A sample is considered correct at
Top-k when any of the first k generated candidates matches the ground-truth
structure under ``StructureMatcher``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import signal
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pymatgen.analysis.structure_matcher import StructureMatcher

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.config import ModelConfig  # noqa: E402
from crystal_language.constrained import (  # noqa: E402
    build_processor,
    formula_from_prompt,
    lattice_block_tokens,
)
from crystal_language.decode import (  # noqa: E402
    ParseError,
    lattice_from_condition_tokens,
    parse_crystal_tokens,
    parse_site_tokens,
    structure_from_sites,
    to_structure,
)
from crystal_language.generate import build_prompt, build_prompt_v2  # noqa: E402
from crystal_language.model import CrystalLM, CrystalSeq2SeqLM, build_model  # noqa: E402
from crystal_language.noise import rasterize_noisy_spectrum  # noqa: E402
from crystal_language.rerank import (  # noqa: E402
    peaks_from_condition_tokens,
    rerank_candidates,
)
from crystal_language.spectra_encoder import rasterize_spectrum  # noqa: E402
from crystal_language.tokenizer import ND_EMB, PXRD_EMB, Tokenizer  # noqa: E402


def composition_counter(parsed: dict[str, Any]) -> Counter[str]:
    return Counter({str(el): int(count) for el, count in parsed["composition"]})


# Degenerate free-lattice candidates (e.g. extremely flat or skewed cells) can
# make pymatgen's Niggli reduction inside StructureMatcher.fit run for hours.
MATCH_TIMEOUT_S = 60
VOL_PER_ATOM_MIN = 1.0
VOL_PER_ATOM_MAX = 200.0


@contextlib.contextmanager
def _time_limit(seconds: int):
    """SIGALRM-based timeout; only safe in the main thread."""

    def _handler(signum, frame):  # noqa: ANN001
        raise TimeoutError(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def candidate_result(
    tokens: list[str],
    target_comp: Counter[str],
    matcher,
    target_structure,
    target_mode: str = "full",
    known_lattice: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Score one candidate; also return a structure payload for reranking."""
    result: dict[str, Any] = {
        "parse_valid": False,
        "composition_match": False,
        "structure_match": False,
        "rmsd": None,
        "n_sites": None,
        "space_group": None,
        "error": None,
    }
    payload: dict[str, Any] | None = None
    try:
        if target_mode == "full":
            parsed = parse_crystal_tokens(tokens)
            result["parse_valid"] = True
            result["n_sites"] = parsed["n_sites"]
            result["space_group"] = parsed["space_group"]
            result["composition_match"] = composition_counter(parsed) == target_comp
            lattice = parsed["lattice"]
            species = [site["element"] for site in parsed["sites"]]
            frac_coords = [site["frac"] for site in parsed["sites"]]
            structure = to_structure(parsed)
        elif target_mode == "sites":
            if known_lattice is None:
                raise ParseError("target_mode='sites' requires known_lattice")
            sites = parse_site_tokens(tokens)
            result["parse_valid"] = True
            result["n_sites"] = len(sites)
            result["space_group"] = None
            result["composition_match"] = Counter(site["element"] for site in sites) == target_comp
            lattice = dict(known_lattice)
            species = [site["element"] for site in sites]
            frac_coords = [site["frac"] for site in sites]
            structure = structure_from_sites(known_lattice, sites)
        else:
            raise ValueError(f"unknown target_mode={target_mode!r}")
        vol_per_atom = float(structure.volume) / max(1, len(structure))
        if not (VOL_PER_ATOM_MIN <= vol_per_atom <= VOL_PER_ATOM_MAX):
            raise ParseError(f"degenerate cell: vol/atom={vol_per_atom:.2f}")
        payload = {"lattice": lattice, "species": species, "frac_coords": frac_coords}
        with _time_limit(MATCH_TIMEOUT_S):
            result["structure_match"] = bool(matcher.fit(structure, target_structure))
        if result["structure_match"]:
            # Normalized RMS displacement (uni3dar / CDVAE protocol metric).
            try:
                with _time_limit(MATCH_TIMEOUT_S):
                    rms = matcher.get_rms_dist(structure, target_structure)
                if rms is not None:
                    result["rmsd"] = round(float(rms[0]), 6)
            except Exception:
                pass
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result, payload


def topk_any(candidates: list[dict[str, Any]], key: str, k: int) -> bool:
    return any(bool(c.get(key)) for c in candidates[:k])


def topk_best_rmsd(candidates: list[dict[str, Any]], k: int) -> float | None:
    """Lowest RMSD among matched candidates in the first k (CDVAE-style RMSE@k)."""
    vals = [
        c["rmsd"]
        for c in candidates[:k]
        if c.get("structure_match") and c.get("rmsd") is not None
    ]
    return min(vals) if vals else None


def generate_candidates(
    model: CrystalLM,
    tokenizer: Tokenizer,
    prompt: list[str],
    num_samples: int,
    samples_per_batch: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
    target_mode: str = "full",
    constrain: str = "none",
    use_lattice: bool = False,
    amp_dtype: torch.dtype | None = None,
    spectra: dict[str, torch.Tensor | None] | None = None,
) -> list[list[str]]:
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

    stop_ids = {tokenizer.eos_id}
    if target_mode == "sites":
        stop_ids.add(tokenizer.token_to_id.get("<ATOM_SITES_END>", tokenizer.eos_id))
    else:
        stop_ids.add(tokenizer.token_to_id.get("<CRYSTAL_END>", tokenizer.eos_id))

    formula = None
    known_lattice_tokens = None
    if constrain == "formula":
        formula = formula_from_prompt(prompt)
        if formula is None:
            raise ValueError(
                "constrain='formula' requires a formula block in the condition prompt"
            )
        if target_mode == "full" and use_lattice and "<LATTICE>" in prompt:
            known_lattice_tokens = lattice_block_tokens(prompt)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (amp_dtype is not None and device != "cpu")
        else contextlib.nullcontext()
    )

    prompt_embeds = None
    cond = None
    if spectra is not None and any(v is not None for v in spectra.values()):
        with autocast_ctx:
            prompt_embeds = model.embed_prompt(prompt_ids, spectra)
            if getattr(model, "cond_adanorm", False):
                batched = {
                    name: (vec.view(1, -1) if vec is not None else None)
                    for name, vec in spectra.items()
                }
                cond = model.encode_condition(batched)

    generated: list[list[str]] = []
    remaining = num_samples
    while remaining > 0:
        n = min(samples_per_batch, remaining)
        processor = None
        if formula is not None:
            processor = build_processor(
                tokenizer=tokenizer,
                formula=formula,
                target_mode=target_mode,
                num_rows=n,
                device=device,
                known_lattice_tokens=known_lattice_tokens,
            )
        with autocast_ctx:
            rows = model.generate_batch(
                prompt_ids if prompt_embeds is None else None,
                num_return=n,
                max_new_tokens=max_new_tokens,
                stop_ids=stop_ids,
                temperature=temperature,
                top_k=top_k,
                processor=processor,
                prompt_embeds=prompt_embeds,
                cond=cond,
            )
        for gen_ids in rows:
            generated.append(tokenizer.decode(gen_ids))
        remaining -= n
    return generated


def generate_seq2seq_candidates(
    model: CrystalSeq2SeqLM,
    tokenizer: Tokenizer,
    prompt: list[str],
    num_samples: int,
    samples_per_batch: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> list[list[str]]:
    encoder_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    encoder_mask = torch.ones_like(encoder_ids, dtype=torch.bool)
    atom_sites_end_id = tokenizer.token_to_id.get("<ATOM_SITES_END>", tokenizer.eos_id)
    generated: list[list[str]] = []
    remaining = num_samples
    while remaining > 0:
        n = min(samples_per_batch, remaining)
        enc = encoder_ids.repeat(n, 1)
        enc_mask = encoder_mask.repeat(n, 1)
        out = torch.full((n, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            ids = out[:, -model.cfg.block_size :]
            logits = model(
                encoder_input_ids=enc,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=ids,
            )["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_id = torch.where(finished, torch.full_like(next_id, tokenizer.eos_id), next_id)
            out = torch.cat([out, next_id[:, None]], dim=1)
            finished |= (next_id == tokenizer.eos_id) | (next_id == atom_sites_end_id)
            if bool(finished.all()):
                break
        for row in out:
            gen_ids = row[1:].tolist()
            generated.append(tokenizer.decode(gen_ids))
        remaining -= n
    return generated


def load_records(path: Path, start_index: int, max_records: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index:
                continue
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_source_index"] = index
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="crystal_language/runs/mp_wbm_v1_block2048")
    parser.add_argument("--checkpoint", default="checkpoint_last.pt")
    parser.add_argument("--data-dir", default="data_prep/outputs/crystal_language_mp_wbm")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--out-dir", default="crystal_language/eval_results/mp_wbm_v1_block2048")
    parser.add_argument("--condition-set", default="pxrd_formula")
    parser.add_argument("--use-lattice", action="store_true")
    parser.add_argument("--target-mode", choices=["full", "sites"], default="full")
    parser.add_argument("--inject-known-lattice", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--top-ks", default="1,20,50")
    parser.add_argument("--samples-per-batch", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k-sampling", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    # Defaults follow the CDVAE / DiffCSP / Uni-3DAR MP-20 protocol so our
    # numbers are directly comparable to the literature. Earlier runs used a
    # stricter 0.2/0.3/5 no-primitive setting that under-reports match rate.
    parser.add_argument("--ltol", type=float, default=0.3)
    parser.add_argument("--stol", type=float, default=0.5)
    parser.add_argument("--angle-tol", type=float, default=10.0)
    parser.add_argument(
        "--primitive-cell",
        dest="primitive_cell",
        action="store_true",
        default=True,
        help="reduce to primitive cell before matching (standard CSP protocol)",
    )
    parser.add_argument(
        "--no-primitive-cell",
        dest="primitive_cell",
        action="store_false",
        help="disable primitive-cell reduction in StructureMatcher",
    )
    parser.add_argument(
        "--constrain",
        choices=["none", "formula"],
        default="none",
        help="formula: force grammar + element quotas from the prompt formula block",
    )
    parser.add_argument("--rerank", action="store_true", help="physical-consistency reranking")
    parser.add_argument("--rerank-workers", type=int, default=4)
    parser.add_argument("--amp", choices=["none", "bf16"], default="none")
    parser.add_argument(
        "--noise-profile",
        choices=["none", "experimental"],
        default="none",
        help="perturb input conditions with experimental artifacts during evaluation",
    )
    parser.add_argument("--noise-seed", type=int, default=20260706,
                        help="base seed for deterministic per-record noisy evaluation")
    parser.add_argument(
        "--rerank-match-sigma",
        type=float,
        default=None,
        help="rasterize σ for PXRD/ND cosine (default: 0.40 when noisy+lattice-free, else 0.25)",
    )
    parser.add_argument(
        "--rerank-shift-max",
        type=float,
        default=None,
        help="max |2θ| shift search in degrees (default: 0.20 when noisy+lattice-free, else 0)",
    )
    parser.add_argument(
        "--store-structures",
        action="store_true",
        help="store parsed candidate structures in the details JSONL",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    top_ks = [k for k in top_ks if k <= args.num_samples]
    if not top_ks:
        raise ValueError("--top-ks must contain at least one value <= --num-samples")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.split}_{args.condition_set}_lat{int(args.use_lattice)}"
        f"_{args.target_mode}_inject{int(args.inject_known_lattice)}"
        f"_cd{int(args.constrain != 'none')}_rr{int(args.rerank)}"
        f"_start{args.start_index}_n{args.max_records or 'all'}_s{args.num_samples}"
    )
    details_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"

    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model_type = ckpt["config"].get("model_type", "decoder")
    model = build_model(
        model_type,
        model_cfg,
        tokenizer.vocab_size,
        tokenizer.pad_id,
        pxrd_emb_id=tokenizer.token_to_id.get(PXRD_EMB),
        nd_emb_id=tokenizer.token_to_id.get(ND_EMB),
        id_to_token=tokenizer.id_to_token,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    arch = getattr(model_cfg, "arch", "v1")
    data_cfg_dump = ckpt["config"].get("data", {})
    spectra_slots = int(data_cfg_dump.get("spectra_slots") or model_cfg.spectra_slots)

    records = load_records(
        Path(args.data_dir) / f"{args.split}.jsonl",
        start_index=args.start_index,
        max_records=args.max_records,
    )
    matcher = StructureMatcher(
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
        primitive_cell=args.primitive_cell,
    )

    counts: Counter[str] = Counter()
    top_counts: dict[int, Counter[str]] = {k: Counter() for k in top_ks}
    top_counts_rr: dict[int, Counter[str]] = {k: Counter() for k in top_ks}
    rmse_sums: dict[int, float] = {k: 0.0 for k in top_ks}
    rmse_counts: dict[int, int] = {k: 0 for k in top_ks}
    rmse_sums_rr: dict[int, float] = {k: 0.0 for k in top_ks}
    rmse_counts_rr: dict[int, int] = {k: 0 for k in top_ks}
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    executor: ProcessPoolExecutor | None = None
    if args.rerank and args.rerank_workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.rerank_workers)

    with details_path.open("w", encoding="utf-8") as details:
        for idx, record in enumerate(records):
            counts["records"] += 1
            try:
                target_parsed = parse_crystal_tokens(record["target"].split())
                target_structure = to_structure(target_parsed)
                target_comp = composition_counter(target_parsed)
            except Exception as exc:
                counts["target_parse_failed"] += 1
                details.write(
                    json.dumps(
                        {"index": idx, "id": record.get("id"), "target_error": str(exc)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            is_v2_record = record.get("rep") == "v2" or arch == "v2"
            noise_rng = random.Random(args.noise_seed + int(record.get("_source_index", idx)))
            spectra_tensors: dict[str, torch.Tensor | None] | None = None
            if is_v2_record:
                prompt, used_spectra = build_prompt_v2(
                    record,
                    args.condition_set,
                    args.use_lattice,
                    spectra_slots,
                    rng=noise_rng,
                    noise_profile=args.noise_profile,
                )
                spec = record.get("spectra") or {}
                spectra_tensors = {}
                for name in ("pxrd", "neutron"):
                    if used_spectra.get(name):
                        xs = spec.get(f"{name}_x") or []
                        ys = spec.get(f"{name}_y") or []
                        if args.noise_profile != "none":
                            vec = rasterize_noisy_spectrum(
                                xs, ys, name=name, rng=noise_rng, profile=args.noise_profile
                            )
                        else:
                            vec = rasterize_spectrum(xs, ys)
                        spectra_tensors[name] = vec.to(device)
                    else:
                        spectra_tensors[name] = None
            else:
                prompt = build_prompt(
                    record,
                    args.condition_set,
                    args.use_lattice,
                    rng=noise_rng,
                    noise_profile=args.noise_profile,
                )
            known_lattice = None
            if args.inject_known_lattice or args.target_mode == "sites":
                known_lattice = lattice_from_condition_tokens(prompt)
            if model_type == "seq2seq":
                gen_tokens = generate_seq2seq_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_samples=args.num_samples,
                    samples_per_batch=args.samples_per_batch,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k_sampling,
                    device=device,
                )
            else:
                gen_tokens = generate_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    num_samples=args.num_samples,
                    samples_per_batch=args.samples_per_batch,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k_sampling,
                    device=device,
                    target_mode=args.target_mode,
                    constrain=args.constrain,
                    use_lattice=args.use_lattice,
                    amp_dtype=amp_dtype,
                    spectra=spectra_tensors,
                )
            results = [
                candidate_result(
                    tokens,
                    target_comp,
                    matcher,
                    target_structure,
                    target_mode=args.target_mode,
                    known_lattice=known_lattice,
                )
                for tokens in gen_tokens
            ]
            candidates = [res for res, _ in results]
            payloads = [payload for _, payload in results]
            for cand in candidates:
                counts["candidates"] += 1
                counts["parse_valid"] += int(cand["parse_valid"])
                counts["composition_match"] += int(cand["composition_match"])
                counts["structure_match"] += int(cand["structure_match"])

            rerank_order: list[int] | None = None
            if args.rerank:
                if is_v2_record:
                    spec = record.get("spectra") or {}
                    ref_pxrd = (
                        list(zip(spec.get("pxrd_x") or [], spec.get("pxrd_y") or []))
                        if spectra_tensors and spectra_tensors.get("pxrd") is not None
                        else None
                    )
                    ref_nd = (
                        list(zip(spec.get("neutron_x") or [], spec.get("neutron_y") or []))
                        if spectra_tensors and spectra_tensors.get("neutron") is not None
                        else None
                    )
                else:
                    ref_pxrd = peaks_from_condition_tokens(prompt, "<PXRD_START>", "<PXRD_END>")
                    ref_nd = peaks_from_condition_tokens(
                        prompt, "<NEUTRON_START>", "<NEUTRON_END>"
                    )
                rr = rerank_candidates(
                    payloads,
                    ref_pxrd,
                    ref_nd,
                    executor=executor,
                    match_sigma=(
                        args.rerank_match_sigma
                        if args.rerank_match_sigma is not None
                        else (0.40 if args.noise_profile != "none" and not args.use_lattice else 0.25)
                    ),
                    shift_max=(
                        args.rerank_shift_max
                        if args.rerank_shift_max is not None
                        else (0.20 if args.noise_profile != "none" and not args.use_lattice else 0.0)
                    ),
                )
                rerank_order = rr.order
                for i, cand in enumerate(candidates):
                    score = rr.scores[i]
                    cand["rerank_score"] = None if score is None else round(score, 4)
                    if rr.detail[i]:
                        cand["rerank"] = rr.detail[i]

            if args.store_structures:
                for cand, payload in zip(candidates, payloads):
                    if payload is not None:
                        cand["structure"] = {
                            "lattice": {
                                k: round(float(v), 4) for k, v in payload["lattice"].items()
                            },
                            "species": payload["species"],
                            "frac_coords": [
                                [round(float(c), 4) for c in xyz]
                                for xyz in payload["frac_coords"]
                            ],
                        }

            row: dict[str, Any] = {
                "index": record.get("_source_index", idx),
                "shard_local_index": idx,
                "id": record.get("id"),
                "target_n_sites": target_parsed["n_sites"],
                "target_composition": sorted(target_comp.items()),
                "prompt_tokens": len(prompt),
                "candidates": candidates,
            }
            if rerank_order is not None:
                row["rerank_order"] = rerank_order
            for k in top_ks:
                top_counts[k]["parse_success"] += int(topk_any(candidates, "parse_valid", k))
                top_counts[k]["composition_success"] += int(
                    topk_any(candidates, "composition_match", k)
                )
                top_counts[k]["structure_success"] += int(
                    topk_any(candidates, "structure_match", k)
                )
                row[f"parse_success@{k}"] = topk_any(candidates, "parse_valid", k)
                row[f"composition_success@{k}"] = topk_any(candidates, "composition_match", k)
                row[f"structure_success@{k}"] = topk_any(candidates, "structure_match", k)
                best = topk_best_rmsd(candidates, k)
                if best is not None:
                    rmse_sums[k] += best
                    rmse_counts[k] += 1
            if rerank_order is not None:
                reranked = [candidates[i] for i in rerank_order]
                for k in top_ks:
                    top_counts_rr[k]["parse_success"] += int(topk_any(reranked, "parse_valid", k))
                    top_counts_rr[k]["composition_success"] += int(
                        topk_any(reranked, "composition_match", k)
                    )
                    top_counts_rr[k]["structure_success"] += int(
                        topk_any(reranked, "structure_match", k)
                    )
                    row[f"rr_structure_success@{k}"] = topk_any(reranked, "structure_match", k)
                    best = topk_best_rmsd(reranked, k)
                    if best is not None:
                        rmse_sums_rr[k] += best
                        rmse_counts_rr[k] += 1

            details.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if (idx + 1) % 10 == 0:
                print(f"evaluated {idx + 1}/{len(records)}", flush=True)

    if executor is not None:
        executor.shutdown(wait=False)

    evaluated = counts["records"] - counts["target_parse_failed"]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "data_dir": args.data_dir,
        "split": args.split,
        "condition_set": args.condition_set,
        "use_lattice": args.use_lattice,
        "target_mode": args.target_mode,
        "inject_known_lattice": args.inject_known_lattice,
        "constrain": args.constrain,
        "rerank": args.rerank,
        "amp": args.amp,
        "num_samples": args.num_samples,
        "top_ks": top_ks,
        "start_index": args.start_index,
        "max_records": args.max_records,
        "temperature": args.temperature,
        "top_k_sampling": args.top_k_sampling,
        "matcher": {
            "ltol": args.ltol,
            "stol": args.stol,
            "angle_tol": args.angle_tol,
            "primitive_cell": args.primitive_cell,
        },
        "counts": dict(counts),
        "candidate_rates": {
            "parse_valid": counts["parse_valid"] / max(1, counts["candidates"]),
            "composition_match": counts["composition_match"] / max(1, counts["candidates"]),
            "structure_match": counts["structure_match"] / max(1, counts["candidates"]),
        },
        "topk_rates": {
            str(k): {
                "parse_success": top_counts[k]["parse_success"] / max(1, evaluated),
                "composition_success": top_counts[k]["composition_success"] / max(1, evaluated),
                "structure_success": top_counts[k]["structure_success"] / max(1, evaluated),
                "rmse": (rmse_sums[k] / rmse_counts[k]) if rmse_counts[k] else None,
                "rmse_n": rmse_counts[k],
            }
            for k in top_ks
        },
        "details_path": str(details_path),
    }
    if args.rerank:
        summary["topk_rates_reranked"] = {
            str(k): {
                "parse_success": top_counts_rr[k]["parse_success"] / max(1, evaluated),
                "composition_success": top_counts_rr[k]["composition_success"]
                / max(1, evaluated),
                "structure_success": top_counts_rr[k]["structure_success"] / max(1, evaluated),
                "rmse": (rmse_sums_rr[k] / rmse_counts_rr[k]) if rmse_counts_rr[k] else None,
                "rmse_n": rmse_counts_rr[k],
            }
            for k in top_ks
        }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
