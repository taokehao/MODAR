"""Sequence dataset assembling condition prefixes and crystal targets.

Each stored record carries the canonical crystal ``target`` plus per-modality
condition blocks. At load time the dataset selects a condition set (optionally
sampled for condition dropout) and a lattice toggle, then assembles a single
autoregressive sequence:

    <BOS> <TASK> solve_structure <COND_START> [blocks] <COND_END> [target] <EOS>

Loss is masked to the target region (prompt labels set to ``-100``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .data.sequence import CONDITION_SETS
from .noise import noisy_peaks_to_di, perturb_condition_block, rasterize_noisy_spectrum
from .peak_encoder import peaks_to_di
from .spectra_encoder import rasterize_spectrum
from .tokenizer import (
    BOS,
    COND_END,
    COND_START,
    EOS,
    ND_EMB,
    PXRD_EMB,
    TASK,
    TASK_NAME,
    Tokenizer,
)

IGNORE_INDEX = -100

SPECTRA_NAMES = ("pxrd", "neutron")
SPECTRA_PLACEHOLDER = {"pxrd": PXRD_EMB, "neutron": ND_EMB}

# Above this file size the dataset switches to lazy mode: it keeps only line
# byte-offsets in memory and seek+parses records on demand (the 6M RealPXRD
# corpus is ~20 GB of JSONL; eager loading per DDP rank would OOM the node).
LAZY_THRESHOLD_BYTES = 2 * 1024**3


class CrystalSequenceDataset(Dataset):
    def __init__(
        self,
        split_path: str | Path,
        tokenizer: Tokenizer,
        cfg: DataConfig,
        training: bool = True,
        lazy: bool | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.training = training
        self.block_size = cfg.block_size
        self.path = Path(split_path)
        self.records: list[dict[str, Any]] = []
        self.skipped_too_long = 0
        self.skipped_small_cell = 0
        self.skipped_empty = 0
        if lazy is None:
            lazy = self.path.exists() and self.path.stat().st_size > LAZY_THRESHOLD_BYTES
        self.lazy = bool(lazy)
        self._offsets: np.ndarray = np.empty(0, dtype=np.int64)
        self._handle = None
        self._handle_pid = -1
        if self.lazy:
            self._offsets, self.skipped_too_long, self.skipped_small_cell = self._load_or_build_index()
            if len(self._offsets) == 0:
                self.skipped_empty = 1
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if self._outside_site_window(record):
                    self.skipped_small_cell += 1
                    continue
                if self._max_sequence_len(record) > self.block_size:
                    self.skipped_too_long += 1
                    continue
                self.records.append(record)
                if cfg.max_records is not None and len(self.records) >= cfg.max_records:
                    break
        if not self.records:
            self.skipped_empty = 1

    # ------------------------------------------------------------------
    # Lazy mode: offset index (cached on disk) + per-process file handle
    # ------------------------------------------------------------------
    def _index_cache_key(self) -> str:
        payload = json.dumps(
            [
                self.cfg.block_size,
                self.cfg.target_mode,
                sorted(self._candidate_condition_sets()),
                self.cfg.use_lattice,
                self.cfg.rep,
                self.cfg.spectra_slots,
                self.cfg.max_records,
                self.cfg.min_sites,
                self.cfg.max_sites,
                getattr(self.cfg, "spectra_encoder", "cnn"),
                getattr(self.cfg, "noise_profile", "none"),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _load_or_build_index(self) -> tuple[np.ndarray, int, int]:
        stat = self.path.stat()
        cache = self.path.parent / f"{self.path.name}.idx-{self._index_cache_key()}.npz"
        if cache.exists():
            data = np.load(cache)
            if (
                int(data["size"]) == stat.st_size
                and int(data["mtime_ns"]) == stat.st_mtime_ns
            ):
                skipped_small = int(data["skipped_small"]) if "skipped_small" in data.files else 0
                return data["offsets"].astype(np.int64), int(data["skipped"]), skipped_small
        offsets: list[int] = []
        skipped = 0
        skipped_small = 0
        pos = 0
        with self.path.open("rb") as fh:
            for raw in fh:
                line_start = pos
                pos += len(raw)
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if self._outside_site_window(record):
                    skipped_small += 1
                    continue
                if self._max_sequence_len(record) > self.block_size:
                    skipped += 1
                    continue
                offsets.append(line_start)
                if self.cfg.max_records is not None and len(offsets) >= self.cfg.max_records:
                    break
        arr = np.asarray(offsets, dtype=np.int64)
        tmp = cache.parent / f"{cache.name}.tmp-{os.getpid()}.npz"
        np.savez(
            tmp,
            offsets=arr,
            skipped=skipped,
            skipped_small=skipped_small,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
        os.replace(tmp, cache)
        return arr, skipped, skipped_small

    def _outside_site_window(self, record: dict[str, Any]) -> bool:
        """True if an n_sites window is set and this record falls outside it.

        Records carry an ``n_sites`` field; missing/None means "keep" so the
        filter never silently drops records that lack the field.
        """
        raw = record.get("n_sites")
        if raw is None:
            return False
        n_sites = int(raw)
        if self.cfg.min_sites is not None and n_sites < int(self.cfg.min_sites):
            return True
        if self.cfg.max_sites is not None and n_sites > int(self.cfg.max_sites):
            return True
        return False

    def _below_min_sites(self, record: dict[str, Any]) -> bool:
        """Alias kept for older call sites / tests."""
        return self._outside_site_window(record)

    def _record(self, index: int) -> dict[str, Any]:
        if not self.lazy:
            return self.records[index]
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            # DataLoader workers fork: each process needs its own handle.
            self._handle = self.path.open("rb")
            self._handle_pid = pid
        self._handle.seek(int(self._offsets[index]))
        return json.loads(self._handle.readline())

    def _target_tokens(self, record: dict[str, Any]) -> list[str]:
        if self.cfg.target_mode == "full":
            return record["target"].split()
        if self.cfg.target_mode != "sites":
            raise ValueError(f"unknown target_mode={self.cfg.target_mode!r}")
        sites = record.get("sites_target")
        if sites:
            return sites.split()
        # Backward-compatible path for already exported JSONL files.
        tokens = record["target"].split()
        start = tokens.index("<ATOM_SITES_START>")
        end = tokens.index("<ATOM_SITES_END>", start) + 1
        return tokens[start:end]

    def __len__(self) -> int:
        return len(self._offsets) if self.lazy else len(self.records)

    def _candidate_condition_sets(self) -> list[str]:
        if self.cfg.condition_set == "mix":
            return list(self.cfg.mix_condition_sets)
        return [self.cfg.condition_set]

    def _candidate_use_lattice(self) -> list[bool]:
        if self.cfg.use_lattice == "always":
            return [True]
        if self.cfg.use_lattice == "never":
            return [False]
        return [False, True]

    def _is_v2(self, record: dict[str, Any]) -> bool:
        return self.cfg.rep == "v2" or record.get("rep") == "v2"

    def _spectrum_available(self, record: dict[str, Any], name: str) -> bool:
        spectra = record.get("spectra") or {}
        return bool(spectra.get(f"{name}_x")) and bool(spectra.get(f"{name}_y"))

    def _prompt_len_for(self, record: dict[str, Any], cond_set: str, use_lattice: bool) -> int:
        cond = record["cond"]
        v2 = self._is_v2(record)
        n = 5  # <BOS> <TASK> solve_structure <COND_START> <COND_END>
        if use_lattice and "lattice" in cond:
            n += len(cond["lattice"].split())
        for name in CONDITION_SETS.get(cond_set, ["pxrd"]):
            if v2 and name in SPECTRA_NAMES:
                if self._spectrum_available(record, name):
                    n += self.cfg.spectra_slots
                continue
            block = cond.get(name)
            if block:
                n += len(block.split())
        return n

    def _max_sequence_len(self, record: dict[str, Any]) -> int:
        target_len = len(self._target_tokens(record)) + 1  # append <EOS> in __getitem__
        max_prompt_len = 0
        for cond_set in self._candidate_condition_sets():
            for use_lattice in self._candidate_use_lattice():
                max_prompt_len = max(
                    max_prompt_len, self._prompt_len_for(record, cond_set, use_lattice)
                )
        return max_prompt_len + target_len

    def _resolve_condition_set(self, rng: random.Random) -> str:
        if self.cfg.condition_set != "mix":
            return self.cfg.condition_set
        names = list(self.cfg.mix_condition_sets)
        weights = self.cfg.mix_condition_weights
        if weights:
            return rng.choices(names, weights=weights, k=1)[0]
        return rng.choice(names)

    def _resolve_use_lattice(self, rng: random.Random) -> bool:
        mode = self.cfg.use_lattice
        if mode == "always":
            return True
        if mode == "never":
            return False
        return rng.random() >= float(getattr(self.cfg, "lattice_free_prob", 0.5))

    def _prompt_tokens(
        self, record: dict[str, Any], rng: random.Random
    ) -> tuple[list[str], dict[str, bool]]:
        cond_set = self._resolve_condition_set(rng)
        block_names = list(CONDITION_SETS.get(cond_set, ["pxrd"]))
        cond = record["cond"]
        v2 = self._is_v2(record)
        used_spectra = {name: False for name in SPECTRA_NAMES}
        tokens: list[str] = [BOS, TASK, TASK_NAME, COND_START]
        if self._resolve_use_lattice(rng) and "lattice" in cond:
            tokens.extend(cond["lattice"].split())
        for name in block_names:
            if v2 and name in SPECTRA_NAMES:
                if self._spectrum_available(record, name):
                    tokens.extend([SPECTRA_PLACEHOLDER[name]] * self.cfg.spectra_slots)
                    used_spectra[name] = True
                continue
            block = cond.get(name)
            if block:
                block_tokens = block.split()
                if self.training and getattr(self.cfg, "noise_profile", "none") != "none":
                    block_tokens = perturb_condition_block(
                        name, block_tokens, rng, self.cfg.noise_profile
                    )
                tokens.extend(block_tokens)
        tokens.append(COND_END)
        return tokens, used_spectra

    def _rasterize(self, record: dict[str, Any], name: str) -> torch.Tensor:
        spectra = record.get("spectra") or {}
        xs = spectra.get(f"{name}_x") or []
        ys = spectra.get(f"{name}_y") or []
        if self.training and getattr(self.cfg, "noise_profile", "none") != "none":
            rng = random.Random(random.getrandbits(63))
            return rasterize_noisy_spectrum(
                xs,
                ys,
                name=name,
                rng=rng,
                profile=self.cfg.noise_profile,
                grid_min=self.cfg.spectra_grid_min,
                grid_max=self.cfg.spectra_grid_max,
                step=self.cfg.spectra_grid_step,
                sigma=self.cfg.spectra_sigma,
            )
        return rasterize_spectrum(
            xs,
            ys,
            grid_min=self.cfg.spectra_grid_min,
            grid_max=self.cfg.spectra_grid_max,
            step=self.cfg.spectra_grid_step,
            sigma=self.cfg.spectra_sigma,
        )

    def _spectrum_feature_shape(self) -> tuple[int, ...]:
        if str(getattr(self.cfg, "spectra_encoder", "cnn")) == "di":
            return (int(getattr(self.cfg, "spectra_max_peaks", 128)), 2)
        grid_len = int(
            round(
                (self.cfg.spectra_grid_max - self.cfg.spectra_grid_min)
                / self.cfg.spectra_grid_step
            )
        ) + 1
        return (grid_len,)

    def _peaks_di(self, record: dict[str, Any], name: str) -> torch.Tensor:
        spectra = record.get("spectra") or {}
        max_peaks = int(getattr(self.cfg, "spectra_max_peaks", 128))
        wavelength = float(getattr(self.cfg, "spectra_wavelength", 1.54184))
        if name == "pxrd":
            native = (record.get("experimental") or {}).get("wavelength_A")
            q_aligned = (record.get("experimental") or {}).get("q_aligned", True)
            if native and not q_aligned:
                wavelength = float(native)
        xs = spectra.get(f"{name}_x") or []
        ys = spectra.get(f"{name}_y") or []
        if self.training and getattr(self.cfg, "noise_profile", "none") != "none":
            rng = random.Random(random.getrandbits(63))
            return noisy_peaks_to_di(
                xs,
                ys,
                name=name,
                rng=rng,
                wavelength=wavelength,
                max_peaks=max_peaks,
                profile=self.cfg.noise_profile,
            )
        return peaks_to_di(xs, ys, wavelength=wavelength, max_peaks=max_peaks)

    def _spectrum_tensor(self, record: dict[str, Any], name: str) -> torch.Tensor:
        if str(getattr(self.cfg, "spectra_encoder", "cnn")) == "di":
            return self._peaks_di(record, name)
        return self._rasterize(record, name)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self._record(index)
        if self.training:
            rng = random.Random(random.getrandbits(63) ^ index)
        else:
            rng = random.Random(index)

        prompt, used_spectra = self._prompt_tokens(record, rng)
        target = self._target_tokens(record)
        target.append(EOS)

        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target)

        ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)

        if len(ids) > self.block_size:
            raise RuntimeError(
                f"sequence length {len(ids)} exceeds block_size={self.block_size}; "
                "dataset filtering should have skipped this record"
            )

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }
        if self._is_v2(record):
            feat_shape = self._spectrum_feature_shape()
            for name in SPECTRA_NAMES:
                if used_spectra[name]:
                    out[f"spectra_{name}"] = self._spectrum_tensor(record, name)
                else:
                    out[f"spectra_{name}"] = torch.zeros(feat_shape, dtype=torch.float32)
        return out


def collate(
    batch: list[dict[str, torch.Tensor]], pad_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention[i, :n] = True
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention}
    if "spectra_pxrd" in batch[0]:
        out["spectra_pxrd"] = torch.stack([item["spectra_pxrd"] for item in batch])
        out["spectra_neutron"] = torch.stack([item["spectra_neutron"] for item in batch])
    return out


def seq2seq_collate(
    batch: list[dict[str, torch.Tensor]], pad_id: int, bos_id: int
) -> dict[str, torch.Tensor]:
    max_prompt_len = max(item["prompt_ids"].numel() for item in batch)
    max_target_len = max(item["target_ids"].numel() for item in batch)
    encoder_input_ids = torch.full((len(batch), max_prompt_len), pad_id, dtype=torch.long)
    encoder_attention_mask = torch.zeros((len(batch), max_prompt_len), dtype=torch.bool)
    decoder_input_ids = torch.full((len(batch), max_target_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_target_len), IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        prompt = item["prompt_ids"]
        target = item["target_ids"]
        prompt_len = prompt.numel()
        target_len = target.numel()
        encoder_input_ids[i, :prompt_len] = prompt
        encoder_attention_mask[i, :prompt_len] = True
        decoder_input_ids[i, 0] = bos_id
        if target_len > 1:
            decoder_input_ids[i, 1:target_len] = target[:-1]
        labels[i, :target_len] = target

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }
