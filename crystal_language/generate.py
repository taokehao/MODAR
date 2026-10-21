#!/usr/bin/env python
"""Generate crystal structures from conditions with a trained crystal LM.

Loads a checkpoint and tokenizer, builds a condition prompt for a chosen
condition set (and optional known lattice) from a sequence JSONL record, then
autoregressively samples a crystal token sequence and parses it. Released CNN
and d-I checkpoints read spectra from the record's ``spectra`` object; spectral
x values are 2-theta in degrees at the supplied wavelength, not d spacings.
The target field is ignored. Sampling does not include physical reranking or
relaxation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.config import ModelConfig  # noqa: E402
from crystal_language.constrained import (  # noqa: E402
    build_processor,
    formula_from_prompt,
    lattice_block_tokens,
)
from crystal_language.data.sequence import CONDITION_SETS  # noqa: E402
from crystal_language.decode import ParseError, parse_crystal_tokens  # noqa: E402
from crystal_language.model import build_model  # noqa: E402
from crystal_language.noise import noisy_peaks_to_di, perturb_condition_block  # noqa: E402
from crystal_language.spectra_encoder import rasterize_spectrum  # noqa: E402
from crystal_language.tokenizer import (  # noqa: E402
    BOS,
    COND_END,
    COND_START,
    ND_EMB,
    PXRD_EMB,
    TASK,
    TASK_NAME,
    Tokenizer,
)


SPECTRA_PLACEHOLDER = {"pxrd": "<PXRD_EMB>", "neutron": "<ND_EMB>"}


def build_prompt(
    record: dict,
    condition_set: str,
    use_lattice: bool,
    rng: random.Random | None = None,
    noise_profile: str = "none",
) -> list[str]:
    cond = record["cond"]
    tokens = [BOS, TASK, TASK_NAME, COND_START]
    if use_lattice and "lattice" in cond:
        tokens.extend(cond["lattice"].split())
    for name in CONDITION_SETS.get(condition_set, ["pxrd"]):
        block = cond.get(name)
        if block:
            block_tokens = block.split()
            if rng is not None and noise_profile != "none":
                block_tokens = perturb_condition_block(name, block_tokens, rng, noise_profile)
            tokens.extend(block_tokens)
    tokens.append(COND_END)
    return tokens


def build_prompt_v2(
    record: dict,
    condition_set: str,
    use_lattice: bool,
    spectra_slots: int,
    rng: random.Random | None = None,
    noise_profile: str = "none",
) -> tuple[list[str], dict[str, bool]]:
    """V2 prompt: spectra blocks become placeholder runs for the dense encoder."""
    cond = record["cond"]
    spectra = record.get("spectra") or {}
    used = {"pxrd": False, "neutron": False}
    tokens = [BOS, TASK, TASK_NAME, COND_START]
    if use_lattice and "lattice" in cond:
        tokens.extend(cond["lattice"].split())
    for name in CONDITION_SETS.get(condition_set, ["pxrd"]):
        if name in SPECTRA_PLACEHOLDER:
            if spectra.get(f"{name}_x") and spectra.get(f"{name}_y"):
                tokens.extend([SPECTRA_PLACEHOLDER[name]] * spectra_slots)
                used[name] = True
            continue
        block = cond.get(name)
        if block:
            block_tokens = block.split()
            if rng is not None and noise_profile != "none":
                block_tokens = perturb_condition_block(name, block_tokens, rng, noise_profile)
            tokens.extend(block_tokens)
    tokens.append(COND_END)
    return tokens, used


def read_record(path: Path, index: int) -> dict:
    """Read a zero-based record from plain or gzip-compressed JSONL."""
    if index < 0:
        raise ValueError("--index must be non-negative")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        current = 0
        for line in handle:
            if not line.strip():
                continue
            if current == index:
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("cond"), dict):
                    raise ValueError("input record requires a cond object")
                if not isinstance(record.get("spectra", {}), dict):
                    raise ValueError("spectra must be an object")
                return record
            current += 1
    raise ValueError(f"index {index} not found in {path}")


def load_model(run_dir: Path, checkpoint: str, device: str):
    """Construct the architecture stored in the checkpoint before loading it."""
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    saved = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    cfg = ModelConfig(**config["model"])
    model_type = config.get("model_type", "decoder")
    if model_type != "decoder":
        raise ValueError("generate.py requires a decoder checkpoint")
    if cfg.arch == "v2" and cfg.spectra_encoder not in {"cnn", "di"}:
        raise ValueError(f"unsupported spectra encoder: {cfg.spectra_encoder}")
    model = build_model(
        model_type, cfg, tokenizer.vocab_size, tokenizer.pad_id,
        pxrd_emb_id=tokenizer.token_to_id.get(PXRD_EMB),
        nd_emb_id=tokenizer.token_to_id.get(ND_EMB),
        id_to_token=tokenizer.id_to_token,
    )
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, tokenizer, config


def prepare_input(record, model, tokenizer, config, args):
    """Build a prompt and the correctly shaped CNN or d-I spectra tensors."""
    cond = record["cond"]
    if args.use_lattice and not cond.get("lattice"):
        raise ValueError("--use-lattice requires a supplied cond.lattice block")
    selected = CONDITION_SETS[args.condition_set]
    for name in selected:
        if name not in SPECTRA_PLACEHOLDER and not cond.get(name):
            raise ValueError(f"condition set requires cond.{name}")
    data_config = config.get("data", {})
    tensors = {"pxrd": None, "neutron": None}
    if model.cfg.arch == "v2":
        slots = int(data_config.get("spectra_slots") or model.cfg.spectra_slots)
        if slots != model.cfg.spectra_slots:
            raise ValueError("checkpoint data/model spectra slot counts disagree")
        prompt, used = build_prompt_v2(
            record, args.condition_set, args.use_lattice, slots
        )
        spec = record.get("spectra", {})
        base_wavelength = float(data_config.get("spectra_wavelength", 1.54184))
        for name in tensors:
            if not used[name]:
                if name in selected:
                    raise ValueError(f"condition set requires non-empty spectra.{name}_x/{name}_y")
                continue
            xs, ys = spec[f"{name}_x"], spec[f"{name}_y"]
            if len(xs) != len(ys) or not all(math.isfinite(float(v)) for v in [*xs, *ys]):
                raise ValueError(f"{name} positions/intensities must have equal length and finite values")
            if not any(float(y) > 0 for y in ys):
                raise ValueError(f"{name} must contain a positive intensity")
            supplied = getattr(args, f"{name}_wavelength", None)
            wavelength = base_wavelength if supplied is None else supplied
            experimental = record.get("experimental") or {}
            if supplied is None and name == "pxrd" and not experimental.get("q_aligned", True):
                if experimental.get("wavelength_A"):
                    wavelength = float(experimental["wavelength_A"])
            if not math.isfinite(wavelength) or wavelength <= 0:
                raise ValueError("wavelengths must be finite and positive")
            if model.cfg.spectra_encoder == "di":
                if not all(0 < float(x) < 180 for x in xs):
                    raise ValueError("d-I conversion requires 2-theta positions between 0 and 180 degrees")
                value = noisy_peaks_to_di(
                    xs, ys, name=name, rng=random.Random(args.seed),
                    wavelength=wavelength, max_peaks=model.cfg.spectra_max_peaks,
                    profile="none",
                )
            else:
                if not math.isclose(wavelength, base_wavelength, rel_tol=1e-6):
                    raise ValueError("CNN inputs must first be remapped to the checkpoint training wavelength")
                value = rasterize_spectrum(
                    xs, ys,
                    grid_min=float(data_config.get("spectra_grid_min", 5.0)),
                    grid_max=float(data_config.get("spectra_grid_max", 80.0)),
                    step=float(data_config.get("spectra_grid_step", 0.05)),
                    sigma=float(data_config.get("spectra_sigma", 0.15)),
                )
            if not bool(torch.isfinite(value).all()) or not bool(value.any()):
                raise ValueError(f"{name} preprocessing produced an empty or non-finite spectrum")
            tensors[name] = value.to(args.device)
    else:
        for name in selected:
            if not cond.get(name):
                raise ValueError(f"condition set requires cond.{name}")
        prompt = build_prompt(record, args.condition_set, args.use_lattice)
    unknown = sorted(set(prompt) - tokenizer.token_to_id.keys())
    if unknown:
        raise ValueError(f"prompt contains tokens absent from this tokenizer: {unknown[:8]}")
    if len(prompt) >= model.cfg.block_size:
        raise ValueError("prompt leaves no room for generation within the context window")
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=args.device)
    embeds = None
    condition = None
    if model.cfg.arch == "v2":
        embeds = model.embed_prompt(ids, tensors)
        # Preserve [B, peaks, 2] for d-I and [B, grid] for CNN encoders.
        batched = {name: value.unsqueeze(0) if value is not None else None
                   for name, value in tensors.items()}
        condition = model.encode_condition(batched)
    return prompt, ids, embeds, condition, tensors


@torch.inference_mode()
def run(args):
    if args.num_samples < 1 or args.max_new_tokens < 1:
        raise ValueError("--num-samples and --max-new-tokens must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0 or args.top_k < 0:
        raise ValueError("--temperature must be finite and positive; --top-k must be non-negative")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    path = args.input_jsonl or Path(args.data_dir) / f"{args.split}.jsonl"
    record = read_record(Path(path), args.index)
    model, tokenizer, config = load_model(Path(args.run_dir), args.checkpoint, args.device)
    prompt, ids, embeds, condition, tensors = prepare_input(record, model, tokenizer, config, args)
    summary = {
        "id": record.get("id"), "condition_set": args.condition_set,
        "use_lattice": args.use_lattice, "prompt_tokens": len(prompt),
        "arch": model.cfg.arch, "device": args.device, "seed": args.seed,
        "constrain": args.constrain, "reranking": False, "relaxation": False,
        "spectra_shapes": {name: list(value.shape) if value is not None else None
                           for name, value in tensors.items()},
    }
    processor = None
    if args.constrain == "formula":
        formula = formula_from_prompt(prompt)
        if not formula:
            raise ValueError("--constrain formula requires a formula prefix")
        processor = build_processor(
            tokenizer=tokenizer, formula=formula, target_mode="full",
            num_rows=args.num_samples, device=args.device,
            known_lattice_tokens=lattice_block_tokens(prompt) if args.use_lattice else None,
        )
    if args.check_only:
        if embeds is None:
            logits = model(ids.unsqueeze(0))["logits"]
        else:
            logits = model(inputs_embeds=embeds.unsqueeze(0), cond=condition)["logits"]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("forward pass produced non-finite logits")
        summary.update(check_only=True, forward_finite=True, logits_shape=list(logits.shape))
        return summary, []
    rows = model.generate_batch(
        ids if embeds is None else None, prompt_embeds=embeds, cond=condition,
        num_return=args.num_samples, max_new_tokens=args.max_new_tokens,
        stop_ids={tokenizer.eos_id, tokenizer.token_to_id.get("<CRYSTAL_END>", tokenizer.eos_id)},
        temperature=args.temperature, top_k=args.top_k or None, processor=processor,
    )
    results = []
    for index, row in enumerate(rows):
        gen_tokens = tokenizer.decode(row)
        result = {"sample": index, "tokens": gen_tokens}
        try:
            parsed = parse_crystal_tokens(gen_tokens)
            result.update(
                valid=True, composition=parsed["composition"], n_sites=parsed["n_sites"],
                space_group=parsed["space_group"], lattice=parsed["lattice"],
                n_parsed_sites=len(parsed["sites"]),
            )
        except ParseError as exc:
            result.update(valid=False, error=str(exc))
        results.append(result)
    return summary, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="dir with tokenizer.json + checkpoint")
    parser.add_argument("--checkpoint", default="checkpoint_last.pt")
    parser.add_argument("--data-dir", default="data_prep/outputs/crystal_language_mp_wbm")
    parser.add_argument("--input-jsonl", type=Path, help="input JSONL or JSONL.gz; overrides --data-dir/--split")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--condition-set", choices=sorted(CONDITION_SETS), default="pxrd_formula")
    parser.add_argument("--use-lattice", action="store_true")
    parser.add_argument("--constrain", choices=["none", "formula"], default="none")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50, help="0 disables sampling top-k")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    parser.add_argument("--pxrd-wavelength", type=float, help="input wavelength in angstrom")
    parser.add_argument("--neutron-wavelength", type=float, help="input wavelength in angstrom")
    parser.add_argument("--check-only", action="store_true", help="load weights and verify one finite forward pass")
    args = parser.parse_args()
    try:
        summary, results = run(args)
        for result in [summary, *results]:
            print(json.dumps(result, ensure_ascii=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
