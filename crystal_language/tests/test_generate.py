"""CPU regression tests for checkpoint-driven generation; no downloads needed.

Run with: python -m unittest discover -s crystal_language/tests -p test_generate.py
"""

import argparse
import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch

from crystal_language.config import ModelConfig
from crystal_language.generate import load_model, prepare_input, read_record, run
from crystal_language.model import build_model, CrystalLM, CrystalLMV2
from crystal_language.tokenizer import (
    ND_EMB, PXRD_EMB, WRAPPER_TOKENS, Tokenizer, angle_token_range,
    coord_token_range, length_token_range, space_group_token_range,
)


class GenerateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tokens = [
            *WRAPPER_TOKENS, PXRD_EMB, ND_EMB, *coord_token_range(),
            *length_token_range(), *angle_token_range(), *space_group_token_range(),
            "<CRYSTAL_START>", "<CRYSTAL_END>", "<FORMULA>", "<N_SITES>",
            "<SPACE_GROUP>", "<LATTICE>", "<ATOM_SITES_START>",
            "<ATOM_SITES_END>", "<ATOM>", "Li", *map(str, range(10)),
            "<PXRD_START>", "<PXRD_END>", ".",
        ]
        self.tokenizer = Tokenizer(list(dict.fromkeys(tokens)))
        self.record = {
            "id": "synthetic-input",
            "cond": {"formula": "<FORMULA> Li 1", "pxrd": "<PXRD_START> 1 <PXRD_END>"},
            "spectra": {"pxrd_x": [20.0, 35.0, 50.0], "pxrd_y": [100.0, 60.0, 20.0]},
            # Inference must not parse or otherwise use this field.
            "target": "deliberately not a crystal target",
        }

    def args(self, **changes):
        values = dict(
            run_dir=str(self.root), checkpoint="checkpoint_last.pt", data_dir=str(self.root),
            input_jsonl=self.root / "test.jsonl", split="test", index=0,
            condition_set="pxrd", use_lattice=False, constrain="none",
            num_samples=1, max_new_tokens=2, temperature=1.0, top_k=5,
            device="cpu", seed=42, threads=1, pxrd_wavelength=None,
            neutron_wavelength=None, check_only=False,
        )
        values.update(changes)
        return argparse.Namespace(**values)

    def save_checkpoint(self, arch="v2", encoder="di"):
        cfg = ModelConfig(
            arch=arch, spectra_encoder=encoder, n_layer=1, n_head=2, n_embd=32,
            dropout=0.0, block_size=128, spectra_slots=2, spectra_max_peaks=8,
            spectra_encoder_layers=1, cond_adanorm=arch == "v2",
            coord_value_emb=arch == "v2",
        )
        config = {"model": asdict(cfg), "data": {"spectra_slots": 2}, "model_type": "decoder"}
        model = build_model(
            "decoder", cfg, self.tokenizer.vocab_size, self.tokenizer.pad_id,
            pxrd_emb_id=self.tokenizer.token_to_id[PXRD_EMB],
            nd_emb_id=self.tokenizer.token_to_id[ND_EMB],
            id_to_token=self.tokenizer.id_to_token,
        )
        self.tokenizer.save(self.root / "tokenizer.json")
        torch.save({"config": config, "model": model.state_dict()}, self.root / "checkpoint_last.pt")
        (self.root / "test.jsonl").write_text(json.dumps(self.record) + "\n")

    def test_cnn_and_di_checkpoints_forward_with_missing_or_present_neutron(self):
        for encoder in ("cnn", "di"):
            with self.subTest(encoder=encoder):
                self.save_checkpoint(encoder=encoder)
                model, tokenizer, config = load_model(self.root, "checkpoint_last.pt", "cpu")
                self.assertIsInstance(model, CrystalLMV2)
                for add_neutron in (False, True):
                    record = copy.deepcopy(self.record)
                    args = self.args()
                    if add_neutron:
                        args.condition_set = "pxrd_neutron"
                        record["spectra"].update(neutron_x=[25.0, 45.0], neutron_y=[50.0, 100.0])
                    with torch.inference_mode():
                        prompt, ids, embeds, condition, tensors = prepare_input(
                            record, model, tokenizer, config, args
                        )
                        expected_shape = (1501,) if encoder == "cnn" else (8, 2)
                        self.assertEqual(tuple(tensors["pxrd"].shape), expected_shape)
                        self.assertEqual(tuple(condition.shape), (1, 32))
                        self.assertEqual(tensors["neutron"] is not None, add_neutron)
                        self.assertEqual(prompt.count(PXRD_EMB), 2)
                        self.assertEqual(prompt.count(ND_EMB), 2 if add_neutron else 0)
                        # A placeholder embedding must be replaced by the observed spectrum.
                        pxrd_index = prompt.index(PXRD_EMB)
                        self.assertFalse(torch.equal(embeds[pxrd_index], model.tok_emb(ids)[pxrd_index]))
                        logits = model(inputs_embeds=embeds.unsqueeze(0), cond=condition)["logits"]
                        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_formula_sampling_is_seeded_and_does_not_read_target(self):
        self.save_checkpoint()
        args = self.args(condition_set="pxrd_formula", constrain="formula", num_samples=2, max_new_tokens=12)
        first = run(args)
        second = run(args)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 2)
        for candidate in first[1]:
            self.assertEqual(candidate["tokens"][:4], ["<CRYSTAL_START>", "<FORMULA>", "Li", "1"])
            self.assertEqual(len(candidate["tokens"]), 12)

    def test_legacy_decoder_still_loads_and_samples(self):
        self.save_checkpoint(arch="v1")
        model, _, _ = load_model(self.root, "checkpoint_last.pt", "cpu")
        self.assertIsInstance(model, CrystalLM)
        summary, results = run(self.args())
        self.assertEqual(summary["arch"], "v1")
        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(results[0]["tokens"]), 2)
        summary, _ = run(self.args(check_only=True))
        self.assertTrue(summary["forward_finite"])

    def test_invalid_or_missing_requested_spectra_fail_explicitly(self):
        self.save_checkpoint()
        model, tokenizer, config = load_model(self.root, "checkpoint_last.pt", "cpu")
        bad_records = []
        for spec in ({}, {"pxrd_x": [20, 30], "pxrd_y": [100]},
                     {"pxrd_x": [float("nan")], "pxrd_y": [100]},
                     {"pxrd_x": [20], "pxrd_y": [0]}):
            bad_records.append({**self.record, "spectra": spec})
        for record in bad_records:
            with self.subTest(spectra=record["spectra"]):
                with self.assertRaises(ValueError):
                    prepare_input(record, model, tokenizer, config, self.args())
        with self.assertRaisesRegex(ValueError, "cond.lattice"):
            prepare_input(self.record, model, tokenizer, config, self.args(use_lattice=True))
        with self.assertRaisesRegex(ValueError, "neutron"):
            prepare_input(self.record, model, tokenizer, config, self.args(condition_set="pxrd_neutron"))
        self.save_checkpoint(encoder="cnn")
        model, tokenizer, config = load_model(self.root, "checkpoint_last.pt", "cpu")
        with self.assertRaisesRegex(ValueError, "remapped"):
            prepare_input(self.record, model, tokenizer, config, self.args(pxrd_wavelength=0.7))

    def test_compressed_record_selection(self):
        path = self.root / "test.jsonl.gz"
        with gzip.open(path, "wt") as handle:
            handle.write("\n" + json.dumps(self.record) + "\n")
        self.assertEqual(read_record(path, 0)["id"], "synthetic-input")
        with self.assertRaises(ValueError):
            read_record(path, -1)
        with self.assertRaises(ValueError):
            read_record(path, 1)

    def test_native_wavelength_metadata_matches_explicit_override(self):
        self.save_checkpoint()
        model, tokenizer, config = load_model(self.root, "checkpoint_last.pt", "cpu")
        record = copy.deepcopy(self.record)
        record["experimental"] = {"q_aligned": False, "wavelength_A": 0.7}
        with torch.inference_mode():
            automatic = prepare_input(record, model, tokenizer, config, self.args())[-1]
            explicit = prepare_input(self.record, model, tokenizer, config,
                                     self.args(pxrd_wavelength=0.7))[-1]
        torch.testing.assert_close(automatic["pxrd"], explicit["pxrd"])


if __name__ == "__main__":
    unittest.main()
