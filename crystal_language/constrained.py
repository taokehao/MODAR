"""Formula-constrained decoding for the crystal language model.

Given the exact formula (from the condition prompt), the grammar of the target
is almost fully determined: structural markers, element tokens, and counts can
be forced, leaving the model free only on the numeric tokens it must actually
predict (coordinates, and in full mode the space group / unknown lattice).

The template is a list of immutable steps shared by all rows of a sampling
batch; each row keeps a lightweight cursor ``(step_idx, n_digits)``.

Step kinds:
    ("fixed", token_id)            exactly this token
    ("choice", frozenset[int])     one token out of a fixed set
    ("varint", min_n, max_n)       1..max_n digit tokens, terminated by the
                                   next step's token (lookahead union)
"""

from __future__ import annotations

from typing import Any

import torch

from .tokenizer import (
    EOS,
    Tokenizer,
    angle_token_range,
    coord_token_range,
    length_token_range,
    space_group_token,
    space_group_token_range,
)

DIGITS = [str(d) for d in range(10)]


def tokenizer_is_v2(tokenizer: Tokenizer) -> bool:
    return "<C000>" in tokenizer.token_to_id


def _ids_for(tokenizer: Tokenizer, tokens: list[str]) -> frozenset[int]:
    table = tokenizer.token_to_id
    ids = frozenset(table[t] for t in tokens if t in table)
    if not ids:
        raise TemplateError("bin token range missing from vocab")
    return ids


class TemplateError(ValueError):
    pass


def _tok(tokenizer: Tokenizer, token: str) -> int:
    tid = tokenizer.token_to_id.get(token)
    if tid is None:
        raise TemplateError(f"token {token!r} missing from vocab")
    return tid


def _digit_ids(tokenizer: Tokenizer) -> frozenset[int]:
    return frozenset(_tok(tokenizer, d) for d in DIGITS)


def _fixed_chars(tokenizer: Tokenizer, text: str) -> list[tuple]:
    return [("fixed", _tok(tokenizer, ch)) for ch in text]


def _coord_steps(tokenizer: Tokenizer) -> list[tuple]:
    """One fractional coordinate: ('0'|'1') '.' dddd."""
    digit_ids = _digit_ids(tokenizer)
    first = frozenset({_tok(tokenizer, "0"), _tok(tokenizer, "1")})
    steps: list[tuple] = [("choice", first), ("fixed", _tok(tokenizer, "."))]
    steps.extend(("choice", digit_ids) for _ in range(4))
    return steps


def _number_steps(tokenizer: Tokenizer, decimals: int, max_int_digits: int = 3) -> list[tuple]:
    """Free positive number: 1..max_int_digits digits, '.', fixed decimals."""
    digit_ids = _digit_ids(tokenizer)
    steps: list[tuple] = [("varint", 1, max_int_digits), ("fixed", _tok(tokenizer, "."))]
    steps.extend(("choice", digit_ids) for _ in range(decimals))
    return steps


def expand_formula(formula: list[tuple[str, int]]) -> list[str]:
    """Sorted site element sequence exactly as the exporter sorts sites."""
    out: list[str] = []
    for element, count in sorted(formula, key=lambda kv: kv[0]):
        out.extend([element] * int(count))
    return out


def lattice_block_tokens(prompt_tokens: list[str]) -> list[str]:
    """Extract the lattice block tokens (V1 char form or V2 bin form) from a prompt."""
    if "<LATTICE>" not in prompt_tokens:
        raise TemplateError("prompt has no <LATTICE> block")
    i = prompt_tokens.index("<LATTICE>")
    out = [prompt_tokens[i]]
    i += 1
    if i < len(prompt_tokens) and prompt_tokens[i].startswith("<L"):
        # V2: exactly three length bins + three angle bins.
        for k in range(6):
            if i >= len(prompt_tokens):
                raise TemplateError("V2 lattice block truncated")
            token = prompt_tokens[i]
            expected = "<L" if k < 3 else "<G"
            if not token.startswith(expected):
                raise TemplateError(f"V2 lattice block malformed at {token!r}")
            out.append(token)
            i += 1
        return out
    num_chars = set("0123456789.-")
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        if i >= len(prompt_tokens) or prompt_tokens[i] != key:
            raise TemplateError(f"lattice block missing {key!r}")
        out.append(prompt_tokens[i])
        i += 1
        n = 0
        while (
            i < len(prompt_tokens)
            and len(prompt_tokens[i]) == 1
            and prompt_tokens[i] in num_chars
        ):
            out.append(prompt_tokens[i])
            i += 1
            n += 1
        if n == 0:
            raise TemplateError(f"lattice value for {key!r} is empty")
    return out


def build_sites_template(tokenizer: Tokenizer, formula: list[tuple[str, int]]) -> list[tuple]:
    steps: list[tuple] = [("fixed", _tok(tokenizer, "<ATOM_SITES_START>"))]
    atom_id = _tok(tokenizer, "<ATOM>")
    if tokenizer_is_v2(tokenizer):
        coord_ids = _ids_for(tokenizer, coord_token_range())
        for element in expand_formula(formula):
            steps.append(("fixed", atom_id))
            steps.append(("fixed", _tok(tokenizer, element)))
            steps.extend(("choice", coord_ids) for _ in range(3))
        steps.append(("fixed", _tok(tokenizer, "<ATOM_SITES_END>")))
        steps.append(("fixed", _tok(tokenizer, EOS)))
        return steps
    axis_ids = {axis: _tok(tokenizer, axis) for axis in ("x", "y", "z")}
    for element in expand_formula(formula):
        steps.append(("fixed", atom_id))
        steps.append(("fixed", _tok(tokenizer, element)))
        for axis in ("x", "y", "z"):
            steps.append(("fixed", axis_ids[axis]))
            steps.extend(_coord_steps(tokenizer))
    steps.append(("fixed", _tok(tokenizer, "<ATOM_SITES_END>")))
    steps.append(("fixed", _tok(tokenizer, EOS)))
    return steps


def build_full_template(
    tokenizer: Tokenizer,
    formula: list[tuple[str, int]],
    known_lattice_tokens: list[str] | None = None,
    space_group: int | None = None,
) -> list[tuple]:
    """Full-target grammar template.

    ``space_group`` pins the space-group token (v2 tokenizer only), which is how
    space-group-stratified sampling forces one branch per group; ``None`` keeps
    the free choice over all 230 groups.
    """
    composition = sorted(formula, key=lambda kv: kv[0])
    n_sites = sum(int(c) for _, c in composition)
    v2 = tokenizer_is_v2(tokenizer)

    steps: list[tuple] = [
        ("fixed", _tok(tokenizer, "<CRYSTAL_START>")),
        ("fixed", _tok(tokenizer, "<FORMULA>")),
    ]
    for element, count in composition:
        steps.append(("fixed", _tok(tokenizer, element)))
        steps.extend(_fixed_chars(tokenizer, str(int(count))))
    steps.append(("fixed", _tok(tokenizer, "<N_SITES>")))
    steps.extend(_fixed_chars(tokenizer, str(n_sites)))
    steps.append(("fixed", _tok(tokenizer, "<SPACE_GROUP>")))
    if v2 and space_group is not None:
        steps.append(("fixed", _tok(tokenizer, space_group_token(int(space_group)))))
    elif v2:
        steps.append(("choice", _ids_for(tokenizer, space_group_token_range())))
    else:
        steps.append(("varint", 1, 3))

    if known_lattice_tokens is not None:
        for token in known_lattice_tokens:
            steps.append(("fixed", _tok(tokenizer, token)))
    elif v2:
        steps.append(("fixed", _tok(tokenizer, "<LATTICE>")))
        length_ids = _ids_for(tokenizer, length_token_range())
        angle_ids = _ids_for(tokenizer, angle_token_range())
        steps.extend(("choice", length_ids) for _ in range(3))
        steps.extend(("choice", angle_ids) for _ in range(3))
    else:
        steps.append(("fixed", _tok(tokenizer, "<LATTICE>")))
        for key in ("a", "b", "c"):
            steps.append(("fixed", _tok(tokenizer, key)))
            steps.extend(_number_steps(tokenizer, decimals=3))
        for key in ("alpha", "beta", "gamma"):
            steps.append(("fixed", _tok(tokenizer, key)))
            steps.extend(_number_steps(tokenizer, decimals=2))

    sites = build_sites_template(tokenizer, formula)
    # sites template ends with <ATOM_SITES_END> <EOS>; full mode inserts <CRYSTAL_END>.
    steps.extend(sites[:-1])
    steps.append(("fixed", _tok(tokenizer, "<CRYSTAL_END>")))
    steps.append(("fixed", _tok(tokenizer, EOS)))
    return steps


def _step_own_ids(step: tuple, digit_ids: frozenset[int]) -> frozenset[int]:
    kind = step[0]
    if kind == "fixed":
        return frozenset({step[1]})
    if kind == "choice":
        return step[1]
    if kind == "varint":
        return digit_ids
    raise TemplateError(f"unknown step kind {kind!r}")


class TemplateCursor:
    """Per-row decoding state over a shared step template."""

    __slots__ = ("template", "digit_ids", "i", "ndigits")

    def __init__(self, template: list[tuple], digit_ids: frozenset[int]) -> None:
        self.template = template
        self.digit_ids = digit_ids
        self.i = 0
        self.ndigits = 0

    @property
    def done(self) -> bool:
        return self.i >= len(self.template)

    def allowed(self) -> frozenset[int]:
        if self.done:
            return frozenset()
        step = self.template[self.i]
        if step[0] == "varint":
            _, min_n, max_n = step
            allowed: set[int] = set()
            if self.ndigits < max_n:
                allowed |= self.digit_ids
            if self.ndigits >= min_n and self.i + 1 < len(self.template):
                allowed |= _step_own_ids(self.template[self.i + 1], self.digit_ids)
            return frozenset(allowed)
        return _step_own_ids(step, self.digit_ids)

    def advance(self, token_id: int) -> None:
        if self.done:
            return
        step = self.template[self.i]
        if step[0] == "varint":
            if token_id in self.digit_ids and self.ndigits < step[2]:
                self.ndigits += 1
                return
            # Terminator token: hand it to the next step.
            self.i += 1
            self.ndigits = 0
            self.advance(token_id)
            return
        self.i += 1
        self.ndigits = 0


class BatchConstraintProcessor:
    """Applies per-row template masks to logits during batched sampling."""

    def __init__(
        self,
        template: list[tuple],
        num_rows: int,
        tokenizer: Tokenizer,
        device: torch.device | str,
        vocab_size: int,
    ) -> None:
        digit_ids = _digit_ids(tokenizer)
        self.cursors = [TemplateCursor(template, digit_ids) for _ in range(num_rows)]
        self.eos_id = tokenizer.eos_id
        self.device = device
        self.vocab_size = vocab_size
        self._mask_cache: dict[frozenset[int], torch.Tensor] = {}
        eos_only = torch.full((vocab_size,), float("-inf"), device=device)
        eos_only[self.eos_id] = 0.0
        self._eos_only = eos_only

    def _mask_row(self, allowed: frozenset[int]) -> torch.Tensor:
        cached = self._mask_cache.get(allowed)
        if cached is None:
            cached = torch.full((self.vocab_size,), float("-inf"), device=self.device)
            cached[list(allowed)] = 0.0
            self._mask_cache[allowed] = cached
        return cached

    def apply(self, logits: torch.Tensor, finished: torch.Tensor) -> torch.Tensor:
        rows = []
        fin = finished.tolist()
        for r, cursor in enumerate(self.cursors):
            if fin[r] or cursor.done:
                rows.append(self._eos_only)
            else:
                allowed = cursor.allowed()
                rows.append(self._mask_row(allowed) if allowed else self._eos_only)
        return logits + torch.stack(rows, dim=0)

    def advance(self, next_ids: torch.Tensor, finished: torch.Tensor) -> None:
        ids = next_ids.tolist()
        fin = finished.tolist()
        for r, cursor in enumerate(self.cursors):
            if not fin[r] and not cursor.done:
                cursor.advance(int(ids[r]))


def build_processor(
    tokenizer: Tokenizer,
    formula: list[tuple[str, int]],
    target_mode: str,
    num_rows: int,
    device: torch.device | str,
    known_lattice_tokens: list[str] | None = None,
    space_group: int | None = None,
) -> BatchConstraintProcessor:
    if target_mode == "sites":
        template = build_sites_template(tokenizer, formula)
    elif target_mode == "full":
        template = build_full_template(tokenizer, formula, known_lattice_tokens, space_group=space_group)
    else:
        raise TemplateError(f"unsupported target_mode {target_mode!r}")
    return BatchConstraintProcessor(
        template=template,
        num_rows=num_rows,
        tokenizer=tokenizer,
        device=device,
        vocab_size=tokenizer.vocab_size,
    )


def formula_from_prompt(prompt_tokens: list[str]) -> list[tuple[str, int]] | None:
    """Parse the exact formula block out of a condition prompt, if present."""
    from .decode import ParseError, formula_from_condition_tokens

    if "<FORMULA>" not in prompt_tokens:
        return None
    try:
        formula = formula_from_condition_tokens(prompt_tokens)
    except ParseError:
        return None
    return formula or None


__all__ = [
    "BatchConstraintProcessor",
    "TemplateCursor",
    "build_full_template",
    "build_processor",
    "build_sites_template",
    "expand_formula",
    "formula_from_prompt",
    "lattice_block_tokens",
]
