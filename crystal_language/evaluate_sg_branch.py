#!/usr/bin/env python
"""Space-group-stratified sampling for the known-lattice, formula-constrained protocol.

Plain sampling lets the decoder spend most of the N samples on its modal space
group; when that mode is wrong the correct group is never proposed, and on large
cells the samples collapse onto a few near-identical arrangements. This driver
reads the model's own distribution over the 230 space-group tokens (one forward
pass over the deterministic prefix), keeps the top-K groups, tempers the
distribution (p^tau) and forces one decoding branch per group with the
constraint grammar. The total sample budget stays at --num-samples, no ground
truth is consulted, and the rest of the pipeline (parsing, matching, physics
rerank, stored structures) is identical to ``evaluate_generation``.

Output rows use the same schema as ``evaluate_generation`` plus ``sg_branches``
(per-group probability and allocation) and per-candidate ``forced_sg``.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.analysis.structure_matcher import StructureMatcher

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.config import ModelConfig  # noqa: E402
from crystal_language.constrained import (  # noqa: E402
    build_full_template,
    build_processor,
    formula_from_prompt,
    lattice_block_tokens,
)
from crystal_language.decode import parse_crystal_tokens, to_structure  # noqa: E402
from crystal_language.evaluate_generation import (  # noqa: E402
    DEFAULT_WAVELENGTH,
    candidate_result,
    composition_counter,
    load_records,
    topk_any,
    topk_best_rmsd,
)
from crystal_language.generate import build_prompt_v2  # noqa: E402
from crystal_language.model import KVCache, build_model  # noqa: E402
from crystal_language.noise import noisy_peaks_to_di, rasterize_noisy_spectrum  # noqa: E402
from crystal_language.rerank import NOISY_SHIFT_STEP, SIGMA, rerank_candidates  # noqa: E402
from crystal_language.tokenizer import ND_EMB, PXRD_EMB, Tokenizer, space_group_token_range  # noqa: E402


def sg_prefix_ids(tokenizer: Tokenizer, formula: list[tuple[str, int]], known_lattice_tokens: list[str] | None) -> list[int]:
    """Token ids the grammar forces before the space-group token (all 'fixed' steps)."""
    ids: list[int] = []
    for step in build_full_template(tokenizer, formula, known_lattice_tokens):
        if step[0] != "fixed":
            break
        ids.append(int(step[1]))
    return ids


@torch.no_grad()
def space_group_distribution(
    model,
    prompt_embeds: torch.Tensor,
    cond: torch.Tensor | None,
    prefix_ids: list[int],
    sg_ids: list[int],
    autocast_ctx,
) -> np.ndarray:
    """p(space group | prompt, forced prefix), normalised over the 230 SG tokens."""
    P = int(prompt_embeds.shape[0])
    cache = KVCache(
        n_layer=model.cfg.n_layer,
        batch_size=1,
        n_head=model.cfg.n_head,
        head_dim=model.cfg.n_embd // model.cfg.n_head,
        max_len=P + len(prefix_ids) + 2,
    )
    with autocast_ctx:
        logits = model.forward(inputs_embeds=prompt_embeds[None], cache=cache, cond=cond)["logits"][0, -1]
        # The cached attention only supports a multi-token block when the cache is
        # empty (prefill), so the forced prefix is fed one token at a time, exactly
        # as the sampler does.
        for tok in prefix_ids:
            step = torch.tensor([[tok]], dtype=torch.long, device=prompt_embeds.device)
            logits = model.forward(step, cache=cache, cond=cond)["logits"][0, -1]
    sg_logits = logits[torch.tensor(sg_ids, device=logits.device)].float()
    return F.softmax(sg_logits, dim=-1).cpu().numpy()


def allocate_branches(
    p: np.ndarray, n_total: int, k_max: int, temper: float, floor: int, p_min: float
) -> list[tuple[int, float, int]]:
    """(space_group, p, n_samples) per branch; tempered p^tau with a per-branch floor."""
    order = np.argsort(-p)
    chosen = [int(i) for i in order[:k_max] if p[i] >= p_min] or [int(order[0])]
    q = np.power(p[chosen], temper)
    q = q / q.sum()
    n = np.floor(q * n_total).astype(int)
    n = np.maximum(n, floor)
    diff = n_total - int(n.sum())
    if diff > 0:
        n[0] += diff
    while diff < 0:
        # take back from the largest branch that still sits above the floor
        j = int(np.argmax(np.where(n > floor, n, -1)))
        if n[j] <= floor:
            break
        n[j] -= 1
        diff += 1
    if int(n.sum()) != n_total:
        # floors exceeded the budget: drop the weakest branches until it fits
        while int(n.sum()) > n_total and len(chosen) > 1:
            chosen.pop()
            n = n[:-1]
        n[0] += n_total - int(n.sum())
    return [(int(chosen[i]) + 1, float(p[chosen[i]]), int(n[i])) for i in range(len(chosen))]


def generate_branch(
    model,
    tokenizer: Tokenizer,
    prompt_ids: torch.Tensor,
    prompt_embeds: torch.Tensor | None,
    cond: torch.Tensor | None,
    formula: list[tuple[str, int]],
    known_lattice_tokens: list[str] | None,
    space_group: int,
    num_samples: int,
    samples_per_batch: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
    autocast_ctx,
) -> list[list[str]]:
    stop_ids = {tokenizer.eos_id, tokenizer.token_to_id.get("<CRYSTAL_END>", tokenizer.eos_id)}
    out: list[list[str]] = []
    remaining = num_samples
    while remaining > 0:
        n = min(samples_per_batch, remaining)
        processor = build_processor(
            tokenizer=tokenizer,
            formula=formula,
            target_mode="full",
            num_rows=n,
            device=device,
            known_lattice_tokens=known_lattice_tokens,
            space_group=space_group,
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
        out.extend(tokenizer.decode(ids) for ids in rows)
        remaining -= n
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_last.pt")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition-set", default="pxrd_formula")
    parser.add_argument("--use-lattice", action="store_true")
    parser.add_argument("--target-mode", choices=["full"], default="full")
    parser.add_argument("--constrain", choices=["formula"], default="formula")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--top-ks", default="1,10,20,50")
    parser.add_argument("--samples-per-batch", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k-sampling", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ltol", type=float, default=0.3)
    parser.add_argument("--stol", type=float, default=0.5)
    parser.add_argument("--angle-tol", type=float, default=10.0)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-workers", type=int, default=2)
    parser.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    parser.add_argument("--noise-profile", choices=["none", "experimental"], default="none")
    parser.add_argument("--noise-seed", type=int, default=20260706)
    parser.add_argument("--store-structures", action="store_true")
    parser.add_argument("--branch-k", type=int, default=12, help="max space groups branched per target")
    parser.add_argument("--branch-temper", type=float, default=0.5, help="allocation ∝ p^tau")
    parser.add_argument("--branch-floor", type=int, default=3, help="min samples per kept branch")
    parser.add_argument("--branch-pmin", type=float, default=0.001, help="drop groups below this probability")
    parser.add_argument(
        "--force-sg-list", default="",
        help="debug: comma-separated space groups branched uniformly instead of the probe allocation",
    )
    args = parser.parse_args()
    forced_list = [int(x) for x in args.force_sg_list.split(",") if x.strip()]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip() and int(x) <= args.num_samples]

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.split}_{args.condition_set}_lat{int(args.use_lattice)}_full_inject0_cd1"
        f"_rr{int(args.rerank)}_start{args.start_index}_n{args.max_records or 'all'}_s{args.num_samples}"
    )
    details_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"
    lock_handle = details_path.with_name(details_path.name + ".lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another eval already holds {details_path}", flush=True)
        return 1

    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model_type = ckpt["config"].get("model_type", "decoder")
    if model_type != "decoder" or getattr(model_cfg, "arch", "v1") != "v2":
        raise SystemExit("evaluate_sg_branch supports the v2 decoder only")
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
    data_cfg_dump = ckpt["config"].get("data", {})
    spectra_slots = int(data_cfg_dump.get("spectra_slots") or model_cfg.spectra_slots)
    spectra_encoder = str(getattr(model_cfg, "spectra_encoder", None) or data_cfg_dump.get("spectra_encoder") or "cnn")
    spectra_max_peaks = int(getattr(model_cfg, "spectra_max_peaks", None) or data_cfg_dump.get("spectra_max_peaks") or 128)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (amp_dtype is not None and device != "cpu")
        else contextlib.nullcontext()
    )
    sg_tokens = space_group_token_range()
    sg_ids = [tokenizer.token_to_id[t] for t in sg_tokens]

    records = load_records(Path(args.data_dir) / f"{args.split}.jsonl", args.start_index, args.max_records)
    matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol, primitive_cell=True)
    executor = ProcessPoolExecutor(max_workers=args.rerank_workers) if args.rerank and args.rerank_workers > 1 else None

    counts: Counter[str] = Counter()
    top_counts: dict[int, Counter[str]] = {k: Counter() for k in top_ks}
    top_counts_rr: dict[int, Counter[str]] = {k: Counter() for k in top_ks}
    branch_stats: Counter[str] = Counter()

    with details_path.open("w", encoding="utf-8") as details:
        for idx, record in enumerate(records):
            counts["records"] += 1
            try:
                target_parsed = parse_crystal_tokens(record["target"].split())
                target_structure = to_structure(target_parsed)
                target_comp = composition_counter(target_parsed)
            except Exception as exc:  # noqa: BLE001
                counts["target_parse_failed"] += 1
                details.write(json.dumps({"index": idx, "id": record.get("id"), "target_error": str(exc)}) + "\n")
                continue

            noise_rng = random.Random(args.noise_seed + int(record.get("_source_index", idx)))
            prompt, used_spectra = build_prompt_v2(
                record, args.condition_set, args.use_lattice, spectra_slots, rng=noise_rng, noise_profile=args.noise_profile
            )
            spec = record.get("spectra") or {}
            spectra_tensors: dict[str, torch.Tensor | None] = {}
            for name in ("pxrd", "neutron"):
                if not used_spectra.get(name):
                    spectra_tensors[name] = None
                    continue
                xs = spec.get(f"{name}_x") or []
                ys = spec.get(f"{name}_y") or []
                if spectra_encoder == "di":
                    spectra_tensors[name] = noisy_peaks_to_di(
                        xs, ys, name=name, rng=noise_rng, wavelength=DEFAULT_WAVELENGTH,
                        max_peaks=spectra_max_peaks, profile=args.noise_profile,
                    ).to(device)
                else:
                    spectra_tensors[name] = rasterize_noisy_spectrum(
                        xs, ys, name=name, rng=noise_rng, profile=args.noise_profile
                    ).to(device)

            formula = formula_from_prompt(prompt)
            if formula is None:
                raise SystemExit("formula block missing from prompt; this driver needs constrain=formula")
            known_lattice_tokens = lattice_block_tokens(prompt) if (args.use_lattice and "<LATTICE>" in prompt) else None

            prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
            prompt_embeds = None
            cond = None
            if any(v is not None for v in spectra_tensors.values()):
                with autocast_ctx:
                    prompt_embeds = model.embed_prompt(prompt_ids, spectra_tensors)
                    if getattr(model, "cond_adanorm", False):
                        batched = {n: (v.unsqueeze(0) if v is not None else None) for n, v in spectra_tensors.items()}
                        cond = model.encode_condition(batched)
            embeds_for_probe = prompt_embeds
            if embeds_for_probe is None:
                with autocast_ctx:
                    embeds_for_probe = model.embed_prompt(prompt_ids, None)

            prefix_ids = sg_prefix_ids(tokenizer, formula, known_lattice_tokens)
            p_sg = space_group_distribution(model, embeds_for_probe, cond, prefix_ids, sg_ids, autocast_ctx)
            if forced_list:
                per = args.num_samples // len(forced_list)
                branches = [(sg, float(p_sg[sg - 1]), per) for sg in forced_list]
            else:
                branches = allocate_branches(
                    p_sg, args.num_samples, args.branch_k, args.branch_temper, args.branch_floor, args.branch_pmin
                )
            branch_stats["branches"] += len(branches)
            branch_stats["records"] += 1

            gen_tokens: list[list[str]] = []
            forced: list[int] = []
            for sg, _p, n in branches:
                if n <= 0:
                    continue
                toks = generate_branch(
                    model, tokenizer, prompt_ids, prompt_embeds, cond, formula, known_lattice_tokens, sg,
                    n, args.samples_per_batch, args.max_new_tokens, args.temperature, args.top_k_sampling,
                    device, autocast_ctx,
                )
                gen_tokens.extend(toks)
                forced.extend([sg] * len(toks))

            results = [
                candidate_result(tokens, target_comp, matcher, target_structure, target_mode="full", known_lattice=None)
                for tokens in gen_tokens
            ]
            candidates = [res for res, _ in results]
            payloads = [payload for _, payload in results]
            for cand, sg in zip(candidates, forced):
                cand["forced_sg"] = sg
                counts["candidates"] += 1
                counts["parse_valid"] += int(cand["parse_valid"])
                counts["composition_match"] += int(cand["composition_match"])
                counts["structure_match"] += int(cand["structure_match"])

            rerank_order = None
            if args.rerank:
                ref_pxrd = (
                    list(zip(spec.get("pxrd_x") or [], spec.get("pxrd_y") or []))
                    if spectra_tensors.get("pxrd") is not None else None
                )
                ref_nd = (
                    list(zip(spec.get("neutron_x") or [], spec.get("neutron_y") or []))
                    if spectra_tensors.get("neutron") is not None else None
                )
                rr = rerank_candidates(
                    payloads, ref_pxrd, ref_nd, executor=executor,
                    match_sigma=SIGMA, shift_max=0.0, shift_step=NOISY_SHIFT_STEP,
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
                            "lattice": {k: round(float(v), 4) for k, v in payload["lattice"].items()},
                            "species": payload["species"],
                            "frac_coords": [[round(float(c), 4) for c in xyz] for xyz in payload["frac_coords"]],
                        }

            top5 = np.argsort(-p_sg)[:5]
            row: dict[str, Any] = {
                "index": record.get("_source_index", idx),
                "shard_local_index": idx,
                "id": record.get("id"),
                "target_n_sites": target_parsed["n_sites"],
                "target_composition": sorted(target_comp.items()),
                "prompt_tokens": len(prompt),
                "sg_branches": [{"sg": sg, "p": round(p, 5), "n": n} for sg, p, n in branches],
                "sg_probe_top": [{"sg": int(i) + 1, "p": round(float(p_sg[i]), 5)} for i in top5],
                "candidates": candidates,
            }
            if rerank_order is not None:
                row["rerank_order"] = rerank_order
            for k in top_ks:
                for key, label in (("parse_valid", "parse_success"), ("composition_match", "composition_success"), ("structure_match", "structure_success")):
                    hit = topk_any(candidates, key, k)
                    top_counts[k][label] += int(hit)
                    row[f"{label}@{k}"] = hit
            if rerank_order is not None:
                reranked = [candidates[i] for i in rerank_order]
                for k in top_ks:
                    hit = topk_any(reranked, "structure_match", k)
                    top_counts_rr[k]["structure_success"] += int(hit)
                    row[f"rr_structure_success@{k}"] = hit
            details.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(
                f"[{idx + 1}/{len(records)}] {record.get('id')} n={target_parsed['n_sites']} branches={len(branches)} "
                f"top_sg={branches[0][0]}(p={branches[0][1]:.2f}) oracle={row.get('structure_success@' + str(top_ks[-1]))}",
                flush=True,
            )

    if executor is not None:
        executor.shutdown(wait=False)
    evaluated = max(1, counts["records"] - counts["target_parse_failed"])
    summary = {
        "driver": "evaluate_sg_branch",
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_k_sampling": args.top_k_sampling,
        "branch": {"k": args.branch_k, "temper": args.branch_temper, "floor": args.branch_floor, "pmin": args.branch_pmin},
        "mean_branches": branch_stats["branches"] / max(1, branch_stats["records"]),
        "counts": dict(counts),
        "topk_rates": {str(k): {lab: top_counts[k][lab] / evaluated for lab in ("parse_success", "composition_success", "structure_success")} for k in top_ks},
        "topk_rates_reranked": {str(k): {"structure_success": top_counts_rr[k]["structure_success"] / evaluated} for k in top_ks},
        "details_path": str(details_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
