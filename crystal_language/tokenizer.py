"""Whitespace tokenizer for the crystal language route.

Sequences are already rendered as space-separated tokens, so tokenization is a
simple split. The tokenizer owns the special wrapper tokens and provides a
deterministic vocabulary built from the exported sequence corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
TASK = "<TASK>"
TASK_NAME = "solve_structure"
COND_START = "<COND_START>"
COND_END = "<COND_END>"

WRAPPER_TOKENS = [PAD, BOS, EOS, UNK, TASK, TASK_NAME, COND_START, COND_END]

# Placeholder tokens whose embeddings are replaced by dense spectra encodings.
PXRD_EMB = "<PXRD_EMB>"
ND_EMB = "<ND_EMB>"

# ---------------------------------------------------------------------------
# V2 bin-token vocabulary (R2 representation)
# ---------------------------------------------------------------------------
# Fractional coordinates: 1/1000 quantization, one token per coordinate.
COORD_BINS = 1000  # <C000> .. <C999>
# Lattice lengths: 0.05 A bins covering 1.00 .. 50.00 A.
LEN_BIN_STEP = 0.05
LEN_BIN_MIN = 20    # 1.00 A
LEN_BIN_MAX = 1000  # 50.00 A
# Lattice angles: 0.5 deg bins covering 30.0 .. 150.0 deg.
ANGLE_BIN_STEP = 0.5
ANGLE_BIN_MIN = 60   # 30.0 deg
ANGLE_BIN_MAX = 300  # 150.0 deg
SPACE_GROUPS = 230   # <SG001> .. <SG230>


def coord_token(value: float) -> str:
    i = int(round((float(value) % 1.0) * COORD_BINS)) % COORD_BINS
    return f"<C{i:03d}>"


def coord_value(token: str) -> float:
    return int(token[2:-1]) / COORD_BINS


def length_token(value: float) -> str:
    i = int(round(float(value) / LEN_BIN_STEP))
    i = min(max(i, LEN_BIN_MIN), LEN_BIN_MAX)
    return f"<L{i:04d}>"


def length_value(token: str) -> float:
    return int(token[2:-1]) * LEN_BIN_STEP


def angle_token(value: float) -> str:
    i = int(round(float(value) / ANGLE_BIN_STEP))
    i = min(max(i, ANGLE_BIN_MIN), ANGLE_BIN_MAX)
    return f"<G{i:03d}>"


def angle_value(token: str) -> float:
    return int(token[2:-1]) * ANGLE_BIN_STEP


def space_group_token(value: int) -> str:
    i = min(max(int(value), 1), SPACE_GROUPS)
    return f"<SG{i:03d}>"


def space_group_value(token: str) -> int:
    return int(token[3:-1])


def coord_token_range() -> list[str]:
    return [f"<C{i:03d}>" for i in range(COORD_BINS)]


def length_token_range() -> list[str]:
    return [f"<L{i:04d}>" for i in range(LEN_BIN_MIN, LEN_BIN_MAX + 1)]


def angle_token_range() -> list[str]:
    return [f"<G{i:03d}>" for i in range(ANGLE_BIN_MIN, ANGLE_BIN_MAX + 1)]


def space_group_token_range() -> list[str]:
    return [f"<SG{i:03d}>" for i in range(1, SPACE_GROUPS + 1)]


class Tokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.id_to_token = list(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}
        for required in (PAD, BOS, EOS, UNK):
            if required not in self.token_to_id:
                raise ValueError(f"vocab missing required token: {required}")
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.unk_id = self.token_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = self.unk_id
        table = self.token_to_id
        return [table.get(tok, unk) for tok in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                out.append(UNK)
            else:
                out.append(self.id_to_token[i])
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"vocab": self.id_to_token}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"])


def build_vocab_from_jsonl(
    jsonl_paths: Iterable[str | Path],
    min_count: int = 1,
    include_bin_tokens: bool = False,
) -> list[str]:
    """Build the vocabulary from exported sequence corpora.

    With ``include_bin_tokens`` the full V2 bin ranges (coordinates, lattice
    lengths/angles, space groups) are pre-registered so that every bin id is
    addressable even if it never occurs in the training split. Data-driven
    tokens (elements, markers, digits) are appended afterwards by frequency.
    """
    counter: Counter[str] = Counter()
    for path in jsonl_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                counter.update(record["target"].split())
                for block in record["cond"].values():
                    counter.update(block.split())

    vocab: list[str] = list(WRAPPER_TOKENS)
    if include_bin_tokens:
        vocab.extend([PXRD_EMB, ND_EMB])
        vocab.extend(coord_token_range())
        vocab.extend(length_token_range())
        vocab.extend(angle_token_range())
        vocab.extend(space_group_token_range())
    seen = set(vocab)
    for token, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < min_count:
            continue
        if token in seen:
            continue
        vocab.append(token)
        seen.add(token)
    return vocab
