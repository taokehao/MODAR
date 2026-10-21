"""Regression checks for CNN and peak-list inputs to generation evaluation."""

from __future__ import annotations

import math
import random
import unittest

import torch

from crystal_language.config import ModelConfig
from crystal_language.evaluate_generation import generate_candidates, prepare_spectra
from crystal_language.model import build_model
from crystal_language.noise import rasterize_noisy_spectrum
from crystal_language.spectra_encoder import rasterize_spectrum
from crystal_language.tokenizer import ND_EMB, PXRD_EMB, WRAPPER_TOKENS, Tokenizer


class EvaluationSpectraTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.record = {
            "spectra": {
                "pxrd_x": [30.0, 60.0],
                "pxrd_y": [100.0, 50.0],
                "neutron_x": [25.0],
                "neutron_y": [100.0],
            }
        }

    def prepare(self, encoder, used=None, noise="none", data_cfg=None):
        return prepare_spectra(
            self.record,
            used or {"pxrd": True, "neutron": True},
            ModelConfig(spectra_encoder=encoder, spectra_max_peaks=8),
            data_cfg or {},
            rng=random.Random(7),
            noise_profile=noise,
            device="cpu",
        )

    def test_di_shape_bragg_conversion_and_missing_neutron(self):
        spectra = self.prepare("di", {"pxrd": True, "neutron": False})
        self.assertEqual(spectra["pxrd"].shape, (8, 2))
        self.assertIsNone(spectra["neutron"])
        expected_d = 1.54184 / (2 * math.sin(math.radians(15)))
        active = spectra["pxrd"][spectra["pxrd"][:, 1] > 0]
        self.assertEqual(len(active), 2)
        self.assertTrue(torch.isclose(active[:, 0], torch.tensor(expected_d)).any())
        self.assertTrue(torch.isfinite(spectra["pxrd"]).all())

    def test_di_uses_native_wavelength_only_for_unaligned_pxrd(self):
        self.record["experimental"] = {"wavelength_A": 0.7, "q_aligned": False}
        native = self.prepare("di")
        self.record["experimental"]["q_aligned"] = True
        aligned = self.prepare("di")
        torch.testing.assert_close(
            native["pxrd"][:, 0], aligned["pxrd"][:, 0] * (0.7 / 1.54184)
        )
        torch.testing.assert_close(native["neutron"], aligned["neutron"])

    def test_cnn_clean_and_noisy_preprocessing_is_unchanged(self):
        for noise in ("none", "experimental"):
            with self.subTest(noise=noise):
                torch.manual_seed(11)
                actual = self.prepare("cnn", noise=noise)
                torch.manual_seed(11)
                rng = random.Random(7)
                for name in ("pxrd", "neutron"):
                    xs = self.record["spectra"][f"{name}_x"]
                    ys = self.record["spectra"][f"{name}_y"]
                    expected = (
                        rasterize_spectrum(xs, ys)
                        if noise == "none"
                        else rasterize_noisy_spectrum(
                            xs, ys, name=name, rng=rng, profile=noise
                        )
                    )
                    self.assertEqual(actual[name].shape, (1501,))
                    torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)

    def test_sampling_preserves_peak_dimension_with_optional_neutron(self):
        tokenizer = Tokenizer(WRAPPER_TOKENS + [PXRD_EMB, ND_EMB, "<CRYSTAL_END>"])
        for encoder in ("di", "cnn"):
            for include_nd in (False, True):
                with self.subTest(encoder=encoder, include_nd=include_nd):
                    cfg = ModelConfig(
                        arch="v2", n_layer=1, n_head=2, n_embd=16,
                        dropout=0, block_size=32, spectra_slots=2,
                        spectra_encoder=encoder, spectra_max_peaks=8,
                        spectra_encoder_layers=1, cond_adanorm=True,
                    )
                    model = build_model(
                        "decoder", cfg, tokenizer.vocab_size, tokenizer.pad_id,
                        pxrd_emb_id=tokenizer.token_to_id[PXRD_EMB],
                        nd_emb_id=tokenizer.token_to_id[ND_EMB],
                        id_to_token=tokenizer.id_to_token,
                    ).eval()
                    prompt = [PXRD_EMB] * 2 + ([ND_EMB] * 2 if include_nd else [])
                    spectra = self.prepare(
                        encoder, {"pxrd": True, "neutron": include_nd}
                    )
                    with torch.no_grad():
                        tokens = generate_candidates(
                            model, tokenizer, prompt, num_samples=1,
                            samples_per_batch=1, max_new_tokens=2, temperature=1,
                            top_k=1, device="cpu", spectra=spectra,
                        )
                    self.assertEqual(len(tokens), 1)
                    self.assertIsInstance(tokens[0], list)
                    self.assertLessEqual(len(tokens[0]), 2)


if __name__ == "__main__":
    unittest.main()
