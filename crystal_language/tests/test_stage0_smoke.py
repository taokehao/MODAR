#!/usr/bin/env python
"""Stage-0 smoke tests: KV-cache equivalence, constrained decoding, reranker.

Run:  python crystal_language/tests/test_stage0_smoke.py \
          [--run-dir crystal_language/runs/mp_wbm_sites_medium_block4096] \
          [--data-dir data_prep/outputs/crystal_language_mp_wbm]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from crystal_language.config import ModelConfig  # noqa: E402
from crystal_language.constrained import (  # noqa: E402
    BatchConstraintProcessor,
    build_full_template,
    build_processor,
    build_sites_template,
    formula_from_prompt,
    lattice_block_tokens,
)
from crystal_language.decode import parse_crystal_tokens, parse_site_tokens  # noqa: E402
from crystal_language.generate import build_prompt, build_prompt_v2  # noqa: E402
from crystal_language.model import CrystalLM, KVCache  # noqa: E402
from crystal_language.rerank import (  # noqa: E402
    peaks_from_condition_tokens,
    rerank_candidates,
)
from crystal_language.tokenizer import Tokenizer  # noqa: E402


def test_kv_cache_equivalence() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(n_layer=2, n_head=4, n_embd=64, dropout=0.0, block_size=64)
    model = CrystalLM(cfg, vocab_size=50, pad_id=0)
    model.eval()

    ids = torch.randint(1, 50, (3, 32))
    with torch.no_grad():
        full_logits = model(ids)["logits"]

        cache = KVCache(cfg.n_layer, 3, cfg.n_head, cfg.n_embd // cfg.n_head, 64)
        prefill_logits = model(ids[:, :20], cache=cache)["logits"]
        step_logits = [prefill_logits[:, -1, :]]
        for t in range(20, 32):
            out = model(ids[:, t : t + 1], cache=cache)["logits"]
            step_logits.append(out[:, -1, :])

    cached = torch.stack(step_logits, dim=1)  # positions 19..31
    reference = full_logits[:, 19:32, :]
    diff = (cached - reference).abs().max().item()
    assert diff < 1e-4, f"KV-cache logits mismatch: {diff}"
    print(f"[ok] kv-cache equivalence  max|diff|={diff:.2e}")


def _random_walk(template, tokenizer: Tokenizer, rng: random.Random) -> list[str]:
    proc = BatchConstraintProcessor(template, 1, tokenizer, "cpu", tokenizer.vocab_size)
    cursor = proc.cursors[0]
    out: list[str] = []
    guard = 0
    while not cursor.done:
        allowed = sorted(cursor.allowed())
        tok = rng.choice(allowed)
        cursor.advance(tok)
        out.append(tokenizer.id_to_token[tok])
        guard += 1
        if guard > 20000:
            raise AssertionError("template walk did not terminate")
    return out


def test_constrained_templates(run_dir: Path, data_dir: Path) -> None:
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    record = None
    with (data_dir / "valid.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            break
    assert record is not None

    prompt = build_prompt(record, "pxrd_formula", use_lattice=True)
    formula = formula_from_prompt(prompt)
    assert formula, "prompt should contain a formula block"
    target_comp = Counter()
    for el, n in formula:
        target_comp[el] += n

    rng = random.Random(123)

    # sites mode
    sites_template = build_sites_template(tokenizer, formula)
    tokens = _random_walk(sites_template, tokenizer, rng)
    sites = parse_site_tokens(tokens)
    comp = Counter(site["element"] for site in sites)
    assert comp == target_comp, f"sites composition mismatch: {comp} != {target_comp}"
    print(f"[ok] sites template walk    n_sites={len(sites)} composition exact")

    # full mode with known lattice
    lat_tokens = lattice_block_tokens(prompt)
    full_template = build_full_template(tokenizer, formula, known_lattice_tokens=lat_tokens)
    tokens = _random_walk(full_template, tokenizer, rng)
    parsed = parse_crystal_tokens(tokens[:-1])  # strip trailing <EOS>
    comp = Counter({el: n for el, n in parsed["composition"]})
    assert comp == target_comp
    ref_lattice = {k: parsed["lattice"][k] for k in ("a", "b", "c")}
    print(
        f"[ok] full template walk     sg={parsed['space_group']} "
        f"lattice_a={ref_lattice['a']} composition exact"
    )

    # full mode with free lattice
    full_free = build_full_template(tokenizer, formula, known_lattice_tokens=None)
    tokens = _random_walk(full_free, tokenizer, rng)
    parsed = parse_crystal_tokens(tokens[:-1])
    assert Counter({el: n for el, n in parsed["composition"]}) == target_comp
    print("[ok] full template walk (free lattice)")

    # batch processor smoke: masked logits keep only allowed entries finite
    proc = build_processor(tokenizer, formula, "sites", 4, "cpu")
    logits = torch.zeros(4, tokenizer.vocab_size)
    finished = torch.zeros(4, dtype=torch.bool)
    masked = proc.apply(logits.clone(), finished)
    finite = torch.isfinite(masked[0]).nonzero().flatten().tolist()
    allowed = sorted(proc.cursors[0].allowed())
    assert finite == allowed, "mask does not match allowed set"
    print("[ok] batch constraint mask")


def test_rerank(run_dir: Path, data_dir: Path) -> None:
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    record = None
    with (data_dir / "valid.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            break
    prompt = build_prompt(record, "pxrd_formula", use_lattice=True)
    ref_pxrd = peaks_from_condition_tokens(prompt, "<PXRD_START>", "<PXRD_END>")
    assert ref_pxrd, "prompt should contain PXRD peaks"

    target = parse_crystal_tokens(record["target"].split())
    truth = {
        "lattice": target["lattice"],
        "species": [s["element"] for s in target["sites"]],
        "frac_coords": [s["frac"] for s in target["sites"]],
    }
    rng = random.Random(7)
    perturbed = {
        "lattice": dict(target["lattice"]),
        "species": list(truth["species"]),
        "frac_coords": [
            [(c + rng.uniform(0.15, 0.4)) % 1.0 for c in xyz] for xyz in truth["frac_coords"]
        ],
    }
    degenerate = {
        "lattice": {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "species": list(truth["species"]),
        "frac_coords": [[0.0, 0.0, 0.0] for _ in truth["frac_coords"]],
    }

    result = rerank_candidates(
        [perturbed, truth, degenerate, None, dict(truth)], ref_pxrd, None, executor=None
    )
    assert result.order[0] in (1, 4), f"ground truth should rank first, got {result.order}"
    assert result.scores[1] is not None and result.scores[2] is not None
    assert result.scores[1] > result.scores[2], "degenerate cell must score below truth"
    assert result.order[-1] == 3, "parse-invalid candidate must sink to the end"
    dup_positions = sorted(result.order.index(i) for i in (1, 4))
    assert dup_positions[1] >= 2, "duplicate of truth should leave the top section"
    print(f"[ok] rerank ordering        order={result.order} score_truth={result.scores[1]:.3f}")


def _build_v2_tokenizer(v2_data_dir: Path) -> "Tokenizer":
    from crystal_language.tokenizer import build_vocab_from_jsonl

    valid = v2_data_dir / "valid.jsonl"
    vocab = build_vocab_from_jsonl([valid], include_bin_tokens=True)
    return Tokenizer(vocab)


def test_v2_roundtrip_and_arch(v2_data_dir: Path) -> None:
    from crystal_language.config import ModelConfig
    from crystal_language.model import CrystalLMV2
    from crystal_language.tokenizer import ND_EMB, PXRD_EMB

    tokenizer = _build_v2_tokenizer(v2_data_dir)
    record = None
    with (v2_data_dir / "valid.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["n_sites"] >= 4:
                break

    # target parses back to the right composition
    parsed = parse_crystal_tokens(record["target"].split())
    sites = parse_site_tokens(record["sites_target"].split())
    assert len(sites) == record["n_sites"], "v2 sites roundtrip lost atoms"
    print(f"[ok] v2 sequence roundtrip  n_sites={len(sites)} sg={parsed['space_group']}")

    # prompt + arch forward + kv-cache + spectra injection + generation
    prompt, used = build_prompt_v2(record, "pxrd_formula", True, spectra_slots=8)
    assert used["pxrd"], "pxrd spectra should be available"
    cfg = ModelConfig(n_layer=2, n_head=4, n_embd=64, dropout=0.0, block_size=512,
                      arch="v2", spectra_slots=8)
    model = CrystalLMV2(
        cfg, tokenizer.vocab_size, tokenizer.pad_id,
        pxrd_emb_id=tokenizer.token_to_id[PXRD_EMB], nd_emb_id=tokenizer.token_to_id.get(ND_EMB),
    )
    model.eval()
    from crystal_language.spectra_encoder import rasterize_spectrum

    spec = record["spectra"]
    vec = rasterize_spectrum(spec["pxrd_x"], spec["pxrd_y"])
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    embeds = model.embed_prompt(ids, {"pxrd": vec, "neutron": None})
    assert embeds.shape[0] == len(prompt)

    # KV-cache equivalence on the v2 arch
    full = model(ids[None])["logits"][0, -1]
    from crystal_language.model import KVCache

    cache = KVCache(cfg.n_layer, 1, cfg.n_head, cfg.n_embd // cfg.n_head, len(prompt))
    inc = model(ids[None], cache=cache)["logits"][0, -1]
    diff = (full - inc).abs().max().item()
    assert diff < 1e-3, f"v2 kv mismatch {diff}"

    formula = formula_from_prompt(prompt)
    rows = model.generate_batch(
        prompt_ids=None, num_return=2, max_new_tokens=400,
        stop_ids={tokenizer.eos_id, tokenizer.token_to_id["<ATOM_SITES_END>"]},
        temperature=1.0, top_k=20,
        processor=build_processor(tokenizer, formula, "sites", 2, "cpu"),
        prompt_embeds=embeds,
    )
    for row in rows:
        toks = tokenizer.decode(row)
        s = parse_site_tokens(toks)
        comp = Counter(x["element"] for x in s)
        want = Counter({el: n for el, n in formula})
        assert comp == want, f"v2 constrained composition {comp} != {want}"
    print("[ok] v2 arch + spectra injection + constrained generation")


def test_coord_value_emb(v2_data_dir: Path) -> None:
    from crystal_language.model import CrystalLMV2, KVCache
    from crystal_language.tokenizer import ND_EMB, PXRD_EMB

    tokenizer = _build_v2_tokenizer(v2_data_dir)
    tid = tokenizer.token_to_id

    torch.manual_seed(0)
    cfg = ModelConfig(n_layer=2, n_head=4, n_embd=64, dropout=0.0, block_size=512,
                      arch="v2", spectra_slots=8, coord_value_emb=True, coord_value_freqs=8)
    model = CrystalLMV2(
        cfg, tokenizer.vocab_size, tokenizer.pad_id,
        pxrd_emb_id=tid[PXRD_EMB], nd_emb_id=tid.get(ND_EMB),
        id_to_token=tokenizer.id_to_token,
    )
    model.eval()

    # 1) kind/value lookup is correct and special tokens stay kind-0
    assert model.num_kind[tid["<C500>"]].item() == 1
    assert abs(model.num_val[tid["<C500>"]].item() - 0.5) < 1e-6
    assert model.num_kind[tid["<L0200>"]].item() == 2  # 10 A
    assert abs(model.num_val[tid["<L0200>"]].item() - (10.0 - 1.0) / 49.0) < 1e-6
    assert model.num_kind[tid["<G180>"]].item() == 3  # 90 deg
    assert abs(model.num_val[tid["<G180>"]].item() - 0.5) < 1e-6
    for tok in ("<CRYSTAL_START>", "<COND_START>", "<LATTICE>", "<SG225>"):
        if tok in tid:
            assert model.num_kind[tid[tok]].item() == 0, f"{tok} wrongly tagged numeric"
    print("[ok] coord-value lookup     C/L/G kinds + values exact, markers kind-0")

    # 2) zero-init => exact no-op vs. the same model with the flag toggled off
    seq = ["<BOS>", "<LATTICE>", "<L0200>", "<L0200>", "<L0200>", "<G180>", "<G180>",
           "<G180>", "<ATOM_SITES_START>", "<ATOM>", "Si", "<C000>", "<C250>", "<C500>"]
    ids = torch.tensor([[tid[t] for t in seq]], dtype=torch.long)
    with torch.no_grad():
        out_on = model(ids)["logits"]
        model.coord_value_emb = False
        out_off = model(ids)["logits"]
        model.coord_value_emb = True
    assert torch.equal(out_on, out_off), "zero-init coord-value emb must be a no-op"
    print("[ok] coord-value zero-init  logits bit-identical with flag off")

    # 3) randomized projections change logits, and KV-cache equivalence holds
    with torch.no_grad():
        for lin in model.numeric_emb.proj:
            torch.nn.init.normal_(lin.weight, std=0.5)
        out_rand = model(ids)["logits"]
        diff_vs_off = (out_rand - out_off).abs().max().item()
        cache = KVCache(cfg.n_layer, 1, cfg.n_head, cfg.n_embd // cfg.n_head, ids.shape[1])
        pre = model(ids[:, :8], cache=cache)["logits"]
        steps = [pre[:, -1]]
        for t in range(8, ids.shape[1]):
            steps.append(model(ids[:, t:t + 1], cache=cache)["logits"][:, -1])
        inc = torch.stack(steps, dim=1)
        kv_diff = (inc - out_rand[:, 7:]).abs().max().item()
    assert diff_vs_off > 1e-3, "value embedding had no effect after randomization"
    assert kv_diff < 1e-3, f"kv mismatch with coord-value emb: {kv_diff}"
    print(f"[ok] coord-value active     effect={diff_vs_off:.3f} kv|diff|={kv_diff:.2e}")

    # 4) zero-init survives self.apply(_init_weights) (AdaLN + numeric clobber fix)
    cfg_ada = ModelConfig(n_layer=2, n_head=4, n_embd=64, dropout=0.0, block_size=512,
                          arch="v2", spectra_slots=8, cond_adanorm=True,
                          coord_value_emb=True, coord_value_freqs=8)
    m2 = CrystalLMV2(cfg_ada, tokenizer.vocab_size, tokenizer.pad_id,
                     pxrd_emb_id=tid[PXRD_EMB], nd_emb_id=tid.get(ND_EMB),
                     id_to_token=tokenizer.id_to_token)
    assert float(m2.blocks[0].adaLN[1].weight.abs().max()) == 0.0, "adaLN zero-init clobbered"
    assert float(m2.numeric_emb.proj[0].weight.abs().max()) == 0.0, "numeric zero-init clobbered"
    print("[ok] zero-init preserved    adaLN + numeric proj stay zero after _init_weights")

    # 5) warm-start: a no-flag checkpoint loads into a flag-on model (only new keys missing)
    cfg_plain = ModelConfig(n_layer=2, n_head=4, n_embd=64, dropout=0.0, block_size=512,
                            arch="v2", spectra_slots=8)
    plain = CrystalLMV2(cfg_plain, tokenizer.vocab_size, tokenizer.pad_id,
                        pxrd_emb_id=tid[PXRD_EMB], nd_emb_id=tid.get(ND_EMB))
    missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all(k.startswith(("numeric_emb.", "num_kind", "num_val")) for k in missing), missing
    print(f"[ok] warm-start compat      missing only new params ({len(missing)} keys)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="crystal_language/runs/mp_wbm_sites_medium_block4096")
    parser.add_argument("--data-dir", default="data_prep/outputs/crystal_language_mp_wbm")
    parser.add_argument("--v2-data-dir", default="data_prep/outputs/crystal_language_mp_wbm_v2")
    args = parser.parse_args()
    run_dir = (PACKAGE_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    data_dir = (PACKAGE_ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    v2_dir = (PACKAGE_ROOT / args.v2_data_dir).resolve() if not Path(args.v2_data_dir).is_absolute() else Path(args.v2_data_dir)

    test_kv_cache_equivalence()
    test_constrained_templates(run_dir, data_dir)
    test_rerank(run_dir, data_dir)
    if (v2_dir / "valid.jsonl").exists():
        test_v2_roundtrip_and_arch(v2_dir)
        test_coord_value_emb(v2_dir)
    else:
        print("[skip] v2 tests (no v2 export yet)")
    print("all smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
