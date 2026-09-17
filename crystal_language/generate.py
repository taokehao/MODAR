#!/usr/bin/env python
"""Generate crystal structures from conditions with a trained crystal LM.

Loads a checkpoint and tokenizer, builds a condition prompt for a chosen
condition set (and optional known lattice) from a sequence JSONL record, then
autoregressively samples a crystal token sequence and parses it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.config import ModelConfig  # noqa: E402
from crystal_language.data.sequence import CONDITION_SETS  # noqa: E402
from crystal_language.decode import ParseError, parse_crystal_tokens  # noqa: E402
from crystal_language.model import CrystalLM  # noqa: E402
from crystal_language.noise import perturb_condition_block  # noqa: E402
from crystal_language.tokenizer import (  # noqa: E402
    BOS,
    COND_END,
    COND_START,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="dir with tokenizer.json + checkpoint")
    parser.add_argument("--checkpoint", default="checkpoint_last.pt")
    parser.add_argument("--data-dir", default="data_prep/outputs/crystal_language_mp_wbm")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--condition-set", default="pxrd_formula")
    parser.add_argument("--use-lattice", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    run_dir = Path(args.run_dir)
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")

    ckpt = torch.load(run_dir / args.checkpoint, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = CrystalLM(model_cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    record_path = Path(args.data_dir) / f"{args.split}.jsonl"
    with record_path.open("r", encoding="utf-8") as handle:
        record = None
        for i, line in enumerate(handle):
            if i == args.index:
                record = json.loads(line)
                break
    if record is None:
        raise SystemExit(f"index {args.index} not found in {record_path}")

    prompt = build_prompt(record, args.condition_set, args.use_lattice)
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    print(json.dumps({"id": record.get("id"), "condition_set": args.condition_set,
                      "use_lattice": args.use_lattice, "prompt_tokens": len(prompt)}))

    for s in range(args.num_samples):
        out = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_id=tokenizer.eos_id,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        gen_ids = out[0, prompt_ids.shape[1]:].tolist()
        gen_tokens = tokenizer.decode(gen_ids)
        try:
            parsed = parse_crystal_tokens(gen_tokens)
            result = {
                "sample": s,
                "valid": True,
                "composition": parsed["composition"],
                "n_sites": parsed["n_sites"],
                "space_group": parsed["space_group"],
                "lattice": parsed["lattice"],
                "n_parsed_sites": len(parsed["sites"]),
            }
        except ParseError as exc:
            result = {"sample": s, "valid": False, "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
