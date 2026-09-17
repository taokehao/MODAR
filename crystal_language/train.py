#!/usr/bin/env python
"""Train the canonical crystal language model on MP+WBM sequences.

This entry point is independent from FlowMM. It reads sequence JSONL produced by
``crystal_language.data.export_sequences`` and writes checkpoints/vocab under its
own run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from crystal_language.dataset import CrystalSequenceDataset, collate, seq2seq_collate  # noqa: E402
from crystal_language.model import build_model  # noqa: E402
from crystal_language.tokenizer import (  # noqa: E402
    ND_EMB,
    PXRD_EMB,
    Tokenizer,
    build_vocab_from_jsonl,
)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * progress))


def get_or_build_tokenizer(data_dir: Path, out_dir: Path, include_bin_tokens: bool) -> Tokenizer:
    vocab_path = out_dir / "tokenizer.json"
    if vocab_path.exists():
        return Tokenizer.load(vocab_path)
    train_path = data_dir / "train.jsonl"
    vocab = build_vocab_from_jsonl([train_path], include_bin_tokens=include_bin_tokens)
    tokenizer = Tokenizer(vocab)
    tokenizer.save(vocab_path)
    return tokenizer


def forward_batch(model, batch: dict[str, torch.Tensor], device: str, model_type: str):
    if model_type == "decoder":
        kwargs = {}
        if "spectra_pxrd" in batch:
            kwargs["spectra"] = {
                "pxrd": batch["spectra_pxrd"].to(device, non_blocking=True),
                "neutron": batch["spectra_neutron"].to(device, non_blocking=True),
            }
        return model(
            batch["input_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            labels=batch["labels"].to(device, non_blocking=True),
            **kwargs,
        )
    if model_type == "seq2seq":
        return model(
            encoder_input_ids=batch["encoder_input_ids"].to(device),
            encoder_attention_mask=batch["encoder_attention_mask"].to(device),
            decoder_input_ids=batch["decoder_input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
    raise ValueError(f"unknown model_type={model_type!r}")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, max_batches: int, model_type: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = forward_batch(model, batch, device, model_type)
        total += float(out["loss"])
        count += 1
    return total / max(1, count)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DataConfig.data_dir)
    parser.add_argument("--out-dir", default=TrainConfig.out_dir)
    parser.add_argument("--condition-set", default=DataConfig.condition_set)
    parser.add_argument("--use-lattice", default=DataConfig.use_lattice)
    parser.add_argument("--target-mode", choices=["full", "sites"], default=DataConfig.target_mode)
    parser.add_argument("--block-size", type=int, default=ModelConfig.block_size)
    parser.add_argument("--model-type", choices=["decoder", "seq2seq"], default="decoder")
    parser.add_argument("--arch", choices=["v1", "v2"], default="v1")
    parser.add_argument("--rep", choices=["v1", "v2"], default=None,
                        help="sequence representation; defaults to --arch")
    parser.add_argument("--n-layer", type=int, default=ModelConfig.n_layer)
    parser.add_argument("--n-encoder-layer", type=int, default=ModelConfig.n_encoder_layer)
    parser.add_argument("--n-head", type=int, default=ModelConfig.n_head)
    parser.add_argument("--n-embd", type=int, default=ModelConfig.n_embd)
    parser.add_argument("--spectra-slots", type=int, default=ModelConfig.spectra_slots)
    parser.add_argument("--spectra-encoder", choices=["cnn", "di"], default=ModelConfig.spectra_encoder)
    parser.add_argument("--spectra-max-peaks", type=int, default=ModelConfig.spectra_max_peaks)
    parser.add_argument("--spectra-encoder-layers", type=int, default=ModelConfig.spectra_encoder_layers)
    parser.add_argument("--cond-adanorm", action="store_true",
                        help="v2: AdaLN/FiLM condition modulation per block")
    parser.add_argument("--cond-drop", type=float, default=ModelConfig.cond_drop,
                        help="condition dropout prob for classifier-free guidance")
    parser.add_argument("--coord-value-emb", action="store_true",
                        help="v2: Fourier value embedding for coord/lattice bin tokens")
    parser.add_argument("--coord-value-freqs", type=int, default=ModelConfig.coord_value_freqs,
                        help="number of Fourier frequencies for coord value embedding")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--grad-accum-steps", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-sites", type=int, default=None,
                        help="keep only records with n_sites >= this value "
                             "(large-cell specialist training; default: no filter)")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="keep only records with n_sites <= this value")
    parser.add_argument("--noise-profile", choices=["none", "experimental"], default="none",
                        help="online condition noise augmentation for realistic spectra/XRF/XPS")
    parser.add_argument("--lattice-free-prob", type=float, default=DataConfig.lattice_free_prob,
                        help="when --use-lattice=mix, probability of a lattice-free prompt")
    parser.add_argument("--mix-condition-weights", default=None,
                        help="optional condition sampling weights for --condition-set=mix; "
                             "either comma-separated numbers aligned with mix_condition_sets "
                             "or name=value entries, e.g. pxrd_xrf=0.25,pxrd_neutron_xrf=0.35")
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default=TrainConfig.amp)
    parser.add_argument("--smoke", action="store_true", help="tiny config for a quick run")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint path to initialize weights from (fine-tuning); "
                             "the run dir must already contain the matching tokenizer.json")
    parser.add_argument("--init-reset-spectra", action="store_true",
                        help="after --init-from, reinitialize the spectra encoder (front-end swap)")
    parser.add_argument("--lazy-data", choices=["auto", "always", "never"], default="auto",
                        help="offset-indexed lazy jsonl loading (auto: files > 2GB)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="build tokenizer + dataset index caches, print stats, exit")
    parser.add_argument("--skip-final-save", action="store_true",
                        help="do not write checkpoint_last.pt (throwaway timing runs)")
    parser.add_argument("--write-step-timing", action="store_true",
                        help="cuda-synchronize each optimizer step and write step_timing.json")
    parser.add_argument("--timing-warmup-steps", type=int, default=0,
                        help="optimizer steps skipped before summarizing s/step")
    parser.add_argument("--timing-label", default=None,
                        help="optional label stored in step_timing.json")
    return parser.parse_args()


def parse_mix_condition_weights(raw: str | None, names: list[str]) -> list[float] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    named = ["=" in p for p in parts]
    if any(named) and not all(named):
        raise ValueError("--mix-condition-weights must be all numbers or all name=value entries")
    if all(named):
        table: dict[str, float] = {}
        for part in parts:
            key, value = part.split("=", 1)
            key = key.strip()
            if key not in names:
                raise ValueError(f"unknown condition-set weight {key!r}")
            table[key] = float(value)
        weights = [float(table.get(name, 0.0)) for name in names]
    else:
        weights = [float(p) for p in parts]
        if len(weights) != len(names):
            raise ValueError(
                f"--mix-condition-weights numeric form must have {len(names)} values"
            )
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("--mix-condition-weights must be non-negative with positive sum")
    return weights


def main() -> int:
    args = build_args()
    if args.rep is None:
        args.rep = args.arch

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if ddp:
        # Generous timeout: rank 0 may spend many minutes building the
        # tokenizer / lazy index caches for large corpora while others wait.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    is_main = rank == 0

    model_cfg = ModelConfig(
        n_layer=args.n_layer,
        n_encoder_layer=args.n_encoder_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        arch=args.arch,
        spectra_slots=args.spectra_slots,
        spectra_encoder=args.spectra_encoder,
        spectra_max_peaks=args.spectra_max_peaks,
        spectra_encoder_layers=args.spectra_encoder_layers,
        cond_adanorm=args.cond_adanorm,
        cond_drop=args.cond_drop,
        coord_value_emb=args.coord_value_emb,
        coord_value_freqs=args.coord_value_freqs,
    )
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        condition_set=args.condition_set,
        use_lattice=args.use_lattice,
        target_mode=args.target_mode,
        block_size=args.block_size,
        max_records=args.max_records,
        min_sites=args.min_sites,
        max_sites=args.max_sites,
        rep=args.rep,
        spectra_slots=args.spectra_slots,
        spectra_encoder=args.spectra_encoder,
        spectra_max_peaks=args.spectra_max_peaks,
        noise_profile=args.noise_profile,
        lattice_free_prob=args.lattice_free_prob,
        mix_condition_weights=parse_mix_condition_weights(
            args.mix_condition_weights, DataConfig().mix_condition_sets
        ),
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        device=device,
        amp=args.amp,
    )
    if args.smoke:
        model_cfg.n_layer = 2
        model_cfg.n_head = 4
        model_cfg.n_embd = 128
        model_cfg.block_size = min(model_cfg.block_size, 512)
        data_cfg.block_size = model_cfg.block_size
        data_cfg.max_records = args.max_records or 256
        train_cfg.batch_size = min(train_cfg.batch_size, 8)
        train_cfg.grad_accum_steps = 1
        train_cfg.max_steps = 20
        train_cfg.warmup_steps = 2
        train_cfg.log_every = 1
        train_cfg.eval_every = 10
        train_cfg.save_every = 20
        train_cfg.num_workers = 0

    data_dir = Path(data_cfg.data_dir)
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rank 0 builds the tokenizer and (in lazy mode) the offset-index caches;
    # the other ranks wait and then load the cached artifacts.
    lazy = {"auto": None, "always": True, "never": False}[args.lazy_data]
    if is_main:
        tokenizer = get_or_build_tokenizer(data_dir, out_dir, include_bin_tokens=args.rep == "v2")
    if ddp:
        dist.barrier()
        tokenizer = Tokenizer.load(out_dir / "tokenizer.json")

    def _build_datasets():
        train = CrystalSequenceDataset(
            data_dir / "train.jsonl", tokenizer, data_cfg, training=True, lazy=lazy
        )
        valid_path = data_dir / "valid.jsonl"
        valid = (
            CrystalSequenceDataset(valid_path, tokenizer, data_cfg, training=False, lazy=lazy)
            if valid_path.exists()
            else None
        )
        return train, valid

    train_ds = valid_ds = None
    if is_main:
        train_ds, valid_ds = _build_datasets()
    if ddp:
        dist.barrier()
        if train_ds is None:
            train_ds, valid_ds = _build_datasets()
    if is_main:
        print(
            "dataset "
            f"train_kept={len(train_ds)} train_skipped_too_long={train_ds.skipped_too_long}"
            f" train_skipped_small_cell={train_ds.skipped_small_cell}",
            flush=True,
        )
        if valid_ds is not None:
            print(
                "dataset "
                f"valid_kept={len(valid_ds)} valid_skipped_too_long={valid_ds.skipped_too_long}"
                f" valid_skipped_small_cell={valid_ds.skipped_small_cell}",
                flush=True,
            )
    if len(train_ds) == 0:
        raise RuntimeError(
            "all training samples were filtered by block_size; increase --block-size "
            "or use a smaller condition/lattice setting"
        )
    if args.prepare_only:
        if is_main:
            print(
                f"prepare-only done: tokenizer + index caches ready "
                f"(train lazy={train_ds.lazy})",
                flush=True,
            )
        if ddp:
            dist.destroy_process_group()
        return 0

    if args.model_type == "decoder":
        collate_fn = partial(collate, pad_id=tokenizer.pad_id)
    else:
        collate_fn = partial(seq2seq_collate, pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id)
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=len(train_ds) >= train_cfg.batch_size,
        pin_memory=device.startswith("cuda"),
    )
    valid_loader = (
        DataLoader(
            valid_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        if valid_ds is not None
        else None
    )

    model = build_model(
        args.model_type,
        model_cfg,
        tokenizer.vocab_size,
        tokenizer.pad_id,
        pxrd_emb_id=tokenizer.token_to_id.get(PXRD_EMB),
        nd_emb_id=tokenizer.token_to_id.get(ND_EMB),
        id_to_token=tokenizer.id_to_token,
    ).to(device)
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location="cpu")
        init_state = init_ckpt.get("model", init_ckpt)
        init_vocab = init_state["tok_emb.weight"].shape[0]
        if init_vocab != tokenizer.vocab_size:
            raise RuntimeError(
                f"--init-from vocab mismatch: checkpoint {init_vocab} vs tokenizer "
                f"{tokenizer.vocab_size}; copy the source run's tokenizer.json into "
                f"the new out dir before launching"
            )
        model.load_state_dict(init_state, strict=False if args.init_reset_spectra else True)
        if args.init_reset_spectra and hasattr(model, "spectra"):
            for module in model.spectra.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        if is_main:
            print(f"initialized weights from {args.init_from} (step {init_ckpt.get('step')})")
        del init_ckpt, init_state
    n_params = sum(p.numel() for p in model.parameters())
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    amp_enabled = train_cfg.amp != "none" and device.startswith("cuda")
    amp_dtype = torch.float16 if train_cfg.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and train_cfg.amp == "fp16")

    config_dump = {
        "model": asdict(model_cfg),
        "model_type": args.model_type,
        "init_from": args.init_from,
        "data": asdict(data_cfg),
        "train": asdict(train_cfg),
        "world_size": world_size,
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "dataset": {
            "train_kept": len(train_ds),
            "train_skipped_too_long": train_ds.skipped_too_long,
            "train_skipped_small_cell": train_ds.skipped_small_cell,
            "valid_kept": len(valid_ds) if valid_ds is not None else None,
            "valid_skipped_too_long": (
                valid_ds.skipped_too_long if valid_ds is not None else None
            ),
            "valid_skipped_small_cell": (
                valid_ds.skipped_small_cell if valid_ds is not None else None
            ),
        },
    }
    if is_main:
        (out_dir / "config.json").write_text(
            json.dumps(config_dump, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(config_dump, indent=2, ensure_ascii=False))

    step_times: list[float] = []
    step = 0
    epoch = 0
    t0 = time.time()
    model.train()
    done = False
    optimizer.zero_grad(set_to_none=True)
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        accum_loss = 0.0
        accum_count = 0
        for batch in train_loader:
            lr = lr_at(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_t0 = time.perf_counter()
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                out = forward_batch(model, batch, device, args.model_type)
                loss = out["loss"] / max(1, train_cfg.grad_accum_steps)
            accum_loss += float(loss.detach()) * max(1, train_cfg.grad_accum_steps)
            accum_count += 1
            scaler.scale(loss).backward()

            if accum_count < train_cfg.grad_accum_steps:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if args.write_step_timing and device.startswith("cuda"):
                torch.cuda.synchronize()
                if step >= args.timing_warmup_steps:
                    step_times.append(time.perf_counter() - step_t0)
            display_loss = accum_loss / max(1, accum_count)
            accum_loss = 0.0
            accum_count = 0

            if is_main and step % train_cfg.log_every == 0:
                dt = time.time() - t0
                print(
                    f"step {step} loss {display_loss:.4f} lr {lr:.2e} "
                    f"elapsed {dt:.1f}s",
                    flush=True,
                )
            if (
                is_main
                and valid_loader is not None
                and step > 0
                and step % train_cfg.eval_every == 0
            ):
                val = evaluate(
                    raw_model, valid_loader, device, max_batches=20, model_type=args.model_type
                )
                print(f"step {step} valid_loss {val:.4f}", flush=True)
                model.train()
            if is_main and step > 0 and step % train_cfg.save_every == 0:
                ckpt = out_dir / f"checkpoint_{step}.pt"
                torch.save(
                    {"model": raw_model.state_dict(), "config": config_dump, "step": step},
                    ckpt,
                )
            step += 1
            if step >= train_cfg.max_steps:
                done = True
                break
        epoch += 1

    if is_main:
        if args.write_step_timing and step_times:
            payload = {
                "label": args.timing_label,
                "optimizer_steps_recorded": len(step_times),
                "median_s_per_step": statistics.median(step_times),
                "mean_s_per_step": statistics.mean(step_times),
            }
            (out_dir / "step_timing.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        if args.skip_final_save:
            print("skip-final-save: not writing checkpoint_last.pt")
        else:
            final = out_dir / "checkpoint_last.pt"
            torch.save(
                {"model": raw_model.state_dict(), "config": config_dump, "step": step}, final
            )
            print(f"saved {final}")
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
