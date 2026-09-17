"""Decoder-only Transformers for crystal language modeling.

Two architectures share one generation path:
  - ``CrystalLM``   (arch "v1"): LayerNorm + learned positions + GELU. Kept
    unchanged so legacy checkpoints keep loading.
  - ``CrystalLMV2`` (arch "v2"): RMSNorm + RoPE + SwiGLU, plus dense spectra
    prefix encoders (PXRD / neutron 1D-CNN -> k prefix embeddings injected at
    `<PXRD_EMB>` / `<ND_EMB>` placeholder positions).
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .peak_encoder import PeakSpectraEncoder
from .spectra_encoder import SpectraEncoder
from .tokenizer import (
    ANGLE_BIN_STEP,
    COORD_BINS,
    LEN_BIN_STEP,
)

IGNORE_INDEX = -100

# Numeric bin-token detection for coordinate-value embedding (V2). Each maps to
# a normalized scalar in [0, 1] alongside a kind code (1 coord / 2 length /
# 3 angle); everything else is kind 0 (no value embedding).
_COORD_RE = re.compile(r"^<C(\d{3})>$")
_LEN_RE = re.compile(r"^<L(\d{4})>$")
_ANG_RE = re.compile(r"^<G(\d{3})>$")
_LEN_NORM = 49.0   # length 1..50 A -> (v - 1) / 49
_ANG_NORM = 120.0  # angle 30..150 deg -> (v - 30) / 120


def _numeric_token_info(tok: str) -> tuple[int, float]:
    m = _COORD_RE.match(tok)
    if m:
        return 1, (int(m.group(1)) / COORD_BINS)
    m = _LEN_RE.match(tok)
    if m:
        return 2, ((int(m.group(1)) * LEN_BIN_STEP - 1.0) / _LEN_NORM)
    m = _ANG_RE.match(tok)
    if m:
        return 3, ((int(m.group(1)) * ANGLE_BIN_STEP - 30.0) / _ANG_NORM)
    return 0, 0.0


class NumericValueEmbedding(nn.Module):
    """Fourier embedding of a per-token numeric value, one head per kind.

    Output projections are zero-initialized so enabling this module is a no-op
    at init (lets it warm-start from a checkpoint without coord-value emb and
    train stably from scratch).
    """

    def __init__(self, n_embd: int, n_freqs: int = 16, n_kinds: int = 3) -> None:
        super().__init__()
        self.n_kinds = n_kinds
        self.register_buffer(
            "freqs", (2.0 ** torch.arange(n_freqs)) * math.pi, persistent=False
        )
        self.proj = nn.ModuleList([nn.Linear(2 * n_freqs, n_embd) for _ in range(n_kinds)])
        for lin in self.proj:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, kinds: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        ang = vals.unsqueeze(-1) * self.freqs  # [B, T, F]
        ff = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, T, 2F]
        out = vals.new_zeros(*kinds.shape, self.proj[0].out_features)
        for k in range(self.n_kinds):
            mask = kinds == (k + 1)
            if bool(mask.any()):
                out[mask] = self.proj[k](ff[mask]).to(out.dtype)
        return out


class KVCache:
    """Pre-allocated per-layer key/value buffers for incremental decoding.

    Buffers are allocated lazily on the first write so that dtype/device follow
    the actual activations (e.g. under autocast).
    """

    def __init__(self, n_layer: int, batch_size: int, n_head: int, head_dim: int, max_len: int) -> None:
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_len = max_len
        self.length = 0
        self._k: list[torch.Tensor | None] = [None] * n_layer
        self._v: list[torch.Tensor | None] = [None] * n_layer

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new k/v at the current offset; return full views up to the new end."""
        B, H, T, D = k.shape
        if self._k[layer] is None:
            self._k[layer] = k.new_empty(self.batch_size, self.n_head, self.max_len, self.head_dim)
            self._v[layer] = v.new_empty(self.batch_size, self.n_head, self.max_len, self.head_dim)
        end = self.length + T
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len}")
        self._k[layer][:, :, self.length : end] = k
        self._v[layer][:, :, self.length : end] = v
        return self._k[layer][:, :, :end], self._v[layer][:, :, :end]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout if self.training else 0.0
        if cache is not None:
            k_all, v_all = cache.write(layer_idx, k, v)
            if T == 1:
                # Single new token attends to the whole cache: no mask needed.
                out = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=dropout_p)
            else:
                # Prefill (cache starts empty): plain causal attention.
                out = F.scaled_dot_product_attention(
                    q, k_all, v_all, dropout_p=dropout_p, is_causal=True
                )
        elif key_padding_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            allowed = causal[None, None, :, :] & key_padding_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=dropout_p
            )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask, cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.ln2(x))
        return x


class GenerationMixin:
    """Shared KV-cache sampling loop for both architectures."""

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids: torch.Tensor | None = None,
        num_return: int = 1,
        max_new_tokens: int = 1024,
        stop_ids: set[int] | None = None,
        temperature: float = 1.0,
        top_k: int | None = 50,
        processor=None,
        pad_id: int | None = None,
        prompt_embeds: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Sample ``num_return`` continuations of a single prompt with KV-cache.

        Either ``prompt_ids`` (1D token ids) or ``prompt_embeds`` ([P, D]
        pre-computed embeddings, e.g. with injected spectra prefixes) must be
        given. ``processor`` may implement ``apply(logits, finished)`` and
        ``advance(next_ids, finished)`` for constrained decoding. Stop tokens
        are included in the returned sequences (prompt excluded).
        """
        self.eval()
        stop_ids = stop_ids or set()
        if prompt_embeds is not None:
            device = prompt_embeds.device
            P = int(prompt_embeds.shape[0])
        else:
            device = prompt_ids.device
            P = int(prompt_ids.numel())
        max_new = min(max_new_tokens, self.cfg.block_size - P)
        if max_new <= 0:
            return [[] for _ in range(num_return)]
        if pad_id is None:
            pad_id = self.pad_id

        cache = KVCache(
            n_layer=self.cfg.n_layer,
            batch_size=num_return,
            n_head=self.cfg.n_head,
            head_dim=self.cfg.n_embd // self.cfg.n_head,
            max_len=P + max_new,
        )
        if cond is not None and cond.shape[0] != num_return:
            cond = cond.expand(num_return, -1).contiguous()
        if prompt_embeds is not None:
            batch_embeds = prompt_embeds[None].expand(num_return, -1, -1).contiguous()
            logits = self.forward(inputs_embeds=batch_embeds, cache=cache, cond=cond)["logits"][:, -1, :]
        else:
            batch = prompt_ids.view(1, -1).expand(num_return, -1).contiguous()
            logits = self.forward(batch, cache=cache, cond=cond)["logits"][:, -1, :]

        finished = torch.zeros(num_return, dtype=torch.bool, device=device)
        stop_tensor = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
        generated = torch.full((num_return, max_new), pad_id, dtype=torch.long, device=device)
        steps = 0
        for t in range(max_new):
            step_logits = logits
            if processor is not None:
                step_logits = processor.apply(step_logits.clone(), finished)
            if temperature != 1.0:
                step_logits = step_logits / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, step_logits.size(-1))
                values, _ = torch.topk(step_logits, k)
                step_logits = step_logits.masked_fill(
                    step_logits < values[:, [-1]], float("-inf")
                )
            probs = F.softmax(step_logits.float(), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_id = torch.where(finished, torch.full_like(next_id, pad_id), next_id)
            generated[:, t] = next_id
            steps = t + 1
            if processor is not None:
                processor.advance(next_id, finished)
            finished |= (next_id[:, None] == stop_tensor[None, :]).any(dim=1)
            if bool(finished.all()) or t == max_new - 1:
                break
            logits = self.forward(next_id[:, None], cache=cache, cond=cond)["logits"][:, -1, :]

        rows: list[list[int]] = []
        gen_cpu = generated[:, :steps].tolist()
        stop_set = set(int(s) for s in stop_ids)
        for row in gen_cpu:
            ids: list[int] = []
            for tok in row:
                if tok == pad_id:
                    break
                ids.append(tok)
                if tok in stop_set:
                    break
            rows.append(ids)
        return rows

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Single-prompt generation kept for backward compatibility."""
        rows = self.generate_batch(
            input_ids.view(-1),
            num_return=input_ids.shape[0] if input_ids.dim() == 2 else 1,
            max_new_tokens=max_new_tokens,
            stop_ids={eos_id},
            temperature=temperature,
            top_k=top_k,
        )
        device = input_ids.device
        base = input_ids.view(1, -1) if input_ids.dim() == 1 else input_ids
        max_len = max(len(r) for r in rows) if rows else 0
        out = []
        for i, r in enumerate(rows):
            padded = r + [eos_id] * (max_len - len(r))
            out.append(torch.cat([base[min(i, base.shape[0] - 1)],
                                  torch.tensor(padded, dtype=torch.long, device=device)]))
        return torch.stack(out, dim=0)


class CrystalLM(GenerationMixin, nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cache: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs_embeds is not None:
            B, T, _ = inputs_embeds.shape
            base = inputs_embeds
            device = inputs_embeds.device
        else:
            B, T = input_ids.shape
            base = self.tok_emb(input_ids)
            device = input_ids.device
        pos_start = cache.length if cache is not None else 0
        if pos_start + T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {pos_start + T} exceeds block size {self.cfg.block_size}"
            )
        pos = torch.arange(pos_start, pos_start + T, device=device)
        x = base + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, attention_mask, cache=cache, layer_idx=layer_idx)
        if cache is not None:
            cache.length += T
        x = self.ln_f(x)
        logits = self.head(x)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            out["loss"] = loss
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D] (already position-sliced)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttentionV2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        dropout_p = self.attn_dropout if self.training else 0.0
        if cache is not None:
            k_all, v_all = cache.write(layer_idx, k, v)
            if T == 1:
                out = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=dropout_p)
            else:
                out = F.scaled_dot_product_attention(
                    q, k_all, v_all, dropout_p=dropout_p, is_causal=True
                )
        elif key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            allowed = causal[None, None, :, :] & key_padding_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FiLM modulation; shift/scale are [B, D] broadcast over the time axis."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class BlockV2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttentionV2(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)
        self.cond_adanorm = bool(getattr(cfg, "cond_adanorm", False))
        if self.cond_adanorm:
            # DiT-style per-block modulation: shift/scale/gate for attn and mlp.
            # Zero-init so the block starts as a plain residual transformer and
            # learns to use conditioning (also lets non-AdaLN ckpts warm-start).
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cfg.n_embd, 6 * cfg.n_embd))
            nn.init.zeros_(self.adaLN[1].weight)
            nn.init.zeros_(self.adaLN[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cond_adanorm and cond is not None:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(cond).chunk(6, dim=-1)
            h = _modulate(self.norm1(x), shift_a, scale_a)
            x = x + gate_a[:, None, :] * self.attn(
                h, key_padding_mask, cos, sin, cache=cache, layer_idx=layer_idx
            )
            h = _modulate(self.norm2(x), shift_m, scale_m)
            x = x + gate_m[:, None, :] * self.mlp(h)
            return x
        x = x + self.attn(self.norm1(x), key_padding_mask, cos, sin, cache=cache, layer_idx=layer_idx)
        x = x + self.mlp(self.norm2(x))
        return x


class CrystalLMV2(GenerationMixin, nn.Module):
    """RMSNorm + RoPE + SwiGLU decoder with dense spectra prefix encoders."""

    def __init__(
        self,
        cfg: ModelConfig,
        vocab_size: int,
        pad_id: int,
        pxrd_emb_id: int | None = None,
        nd_emb_id: int | None = None,
        id_to_token: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.pxrd_emb_id = pxrd_emb_id
        self.nd_emb_id = nd_emb_id
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([BlockV2(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        if str(getattr(cfg, "spectra_encoder", "cnn")) == "di":
            self.spectra = PeakSpectraEncoder(
                cfg.n_embd,
                cfg.spectra_slots,
                n_layer=int(getattr(cfg, "spectra_encoder_layers", 3)),
                n_head=cfg.n_head,
            )
        else:
            self.spectra = SpectraEncoder(cfg.n_embd, cfg.spectra_slots)
        self.cond_adanorm = bool(getattr(cfg, "cond_adanorm", False))
        self.cond_drop = float(getattr(cfg, "cond_drop", 0.0))
        if self.cond_adanorm:
            # Pool the dense PXRD/neutron encodings into one global condition
            # vector that AdaLN-modulates every block. A learned null vector is
            # the classifier-free-guidance "unconditional" embedding.
            n_mod = len(SpectraEncoder.MODALITIES)
            self.cond_proj = nn.Sequential(
                nn.Linear(n_mod * cfg.n_embd, cfg.n_embd),
                nn.SiLU(),
                nn.Linear(cfg.n_embd, cfg.n_embd),
            )
            self.cond_null = nn.Parameter(torch.zeros(cfg.n_embd))
        self.coord_value_emb = bool(getattr(cfg, "coord_value_emb", False))
        if self.coord_value_emb:
            # Persistent buffers so the value lookup is restored on ckpt load
            # even when id_to_token is not supplied at eval time.
            kinds = torch.zeros(vocab_size, dtype=torch.long)
            vals = torch.zeros(vocab_size, dtype=torch.float32)
            if id_to_token is not None:
                for i, tok in enumerate(id_to_token):
                    k, v = _numeric_token_info(tok)
                    kinds[i] = k
                    vals[i] = v
            self.register_buffer("num_kind", kinds, persistent=True)
            self.register_buffer("num_val", vals, persistent=True)
            self.numeric_emb = NumericValueEmbedding(
                cfg.n_embd, int(getattr(cfg, "coord_value_freqs", 16))
            )
        self.apply(self._init_weights)
        # self.apply above re-inits every nn.Linear with normal(0.02), which
        # would clobber the deliberate zero-init of the AdaLN modulation and the
        # numeric-value projections (they must start as no-ops). Restore them.
        for block in self.blocks:
            if getattr(block, "cond_adanorm", False):
                nn.init.zeros_(block.adaLN[1].weight)
                nn.init.zeros_(block.adaLN[1].bias)
        if self.coord_value_emb:
            for lin in self.numeric_emb.proj:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        head_dim = cfg.n_embd // cfg.n_head
        inv_freq = 1.0 / (
            cfg.rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(cfg.block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _inject_spectra(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        spectra: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        for name, tok_id in (("pxrd", self.pxrd_emb_id), ("neutron", self.nd_emb_id)):
            vec = spectra.get(name)
            if vec is None or tok_id is None:
                continue
            mask = input_ids == tok_id
            rows = mask.any(dim=1)
            if not bool(rows.any()):
                continue
            enc = self.spectra(name, vec[rows].to(x.dtype))
            x = torch.masked_scatter(
                x, mask[:, :, None], enc.reshape(-1, x.size(-1)).to(x.dtype)
            )
        return x

    def _add_numeric(self, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Add the Fourier value embedding of coordinate / lattice bin tokens."""
        if not self.coord_value_emb or input_ids is None:
            return x
        kinds = self.num_kind[input_ids]
        vals = self.num_val[input_ids]
        return x + self.numeric_emb(kinds, vals).to(x.dtype)

    def encode_condition(
        self, spectra: dict[str, torch.Tensor | None] | None
    ) -> torch.Tensor | None:
        """Pool the dense spectra encodings into one condition vector [B, D]."""
        if not self.cond_adanorm or not spectra:
            return None
        ref = next((v for v in spectra.values() if v is not None), None)
        if ref is None:
            return None
        B = ref.shape[0]
        device = ref.device
        dtype = self.cond_null.dtype
        feat_shape = tuple(ref.shape[1:])
        parts: list[torch.Tensor] = []
        for name in SpectraEncoder.MODALITIES:
            vec = spectra.get(name)
            # An absent modality is encoded from an all-zero input, the exact
            # representation the dataset feeds at train time (it fills unused
            # modalities with zeros), so train/inference stay aligned.
            if vec is None:
                vec = torch.zeros(B, *feat_shape, device=device, dtype=dtype)
            enc = self.spectra(name, vec.to(dtype))  # [B, k, D]
            parts.append(enc.mean(dim=1))
        return self.cond_proj(torch.cat(parts, dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cache: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spectra: dict[str, torch.Tensor | None] | None = None,
        cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs_embeds is not None:
            B, T, _ = inputs_embeds.shape
            x = inputs_embeds
            device = inputs_embeds.device
        else:
            B, T = input_ids.shape
            x = self.tok_emb(input_ids)
            device = input_ids.device
            x = self._add_numeric(x, input_ids)
            if spectra:
                x = self._inject_spectra(x, input_ids, spectra)
        if self.cond_adanorm and cond is None:
            cond = self.encode_condition(spectra)
            if cond is not None and self.training and self.cond_drop > 0:
                drop = torch.rand(cond.shape[0], 1, device=cond.device) < self.cond_drop
                cond = torch.where(drop, self.cond_null.to(cond.dtype)[None, :], cond)
            if cond is None:
                # AdaLN is always-on once enabled: an unconditioned forward uses
                # the learned null vector (same path the CFG drop trains), never
                # the ungated branch, so train/inference stay consistent.
                cond = self.cond_null.to(x.dtype)[None, :].expand(B, -1)
        pos_start = cache.length if cache is not None else 0
        if pos_start + T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {pos_start + T} exceeds block size {self.cfg.block_size}"
            )
        cos = self.rope_cos[pos_start : pos_start + T]
        sin = self.rope_sin[pos_start : pos_start + T]

        x = self.drop(x)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, attention_mask, cos, sin, cache=cache, layer_idx=layer_idx, cond=cond)
        if cache is not None:
            cache.length += T
        x = self.norm_f(x)
        logits = self.head(x)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def embed_prompt(
        self, prompt_ids: torch.Tensor, spectra: dict[str, torch.Tensor | None] | None
    ) -> torch.Tensor:
        """Token embeddings for one prompt with spectra prefixes injected: [P, D]."""
        ids = prompt_ids.view(1, -1)
        x = self.tok_emb(ids)
        x = self._add_numeric(x, ids)
        if spectra:
            batched = {
                name: (vec.unsqueeze(0) if vec is not None else None)
                for name, vec in spectra.items()
            }
            x = self._inject_spectra(x, ids, batched)
        return x[0]


def build_model(
    model_type: str,
    cfg: ModelConfig,
    vocab_size: int,
    pad_id: int,
    pxrd_emb_id: int | None = None,
    nd_emb_id: int | None = None,
    id_to_token: list[str] | None = None,
) -> nn.Module:
    if model_type == "seq2seq":
        return CrystalSeq2SeqLM(cfg, vocab_size, pad_id)
    if getattr(cfg, "arch", "v1") == "v2":
        return CrystalLMV2(
            cfg,
            vocab_size,
            pad_id,
            pxrd_emb_id=pxrd_emb_id,
            nd_emb_id=nd_emb_id,
            id_to_token=id_to_token,
        )
    return CrystalLM(cfg, vocab_size, pad_id)


class CrystalSeq2SeqLM(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.encoder_pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.decoder_pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.n_embd,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.n_embd,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        n_encoder_layer = cfg.n_encoder_layer or max(1, cfg.n_layer // 2)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layer)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, S = encoder_input_ids.shape
        _, T = decoder_input_ids.shape
        if S > self.cfg.block_size or T > self.cfg.block_size:
            raise ValueError(
                f"sequence length exceeds block size: encoder={S} decoder={T} "
                f"block_size={self.cfg.block_size}"
            )
        enc_pos = torch.arange(S, device=encoder_input_ids.device)
        dec_pos = torch.arange(T, device=decoder_input_ids.device)
        enc = self.tok_emb(encoder_input_ids) + self.encoder_pos_emb(enc_pos)[None, :, :]
        dec = self.tok_emb(decoder_input_ids) + self.decoder_pos_emb(dec_pos)[None, :, :]
        enc = self.drop(enc)
        dec = self.drop(dec)

        src_key_padding_mask = None
        if encoder_attention_mask is not None:
            src_key_padding_mask = ~encoder_attention_mask
        memory = self.encoder(enc, src_key_padding_mask=src_key_padding_mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=decoder_input_ids.device
        )
        out = self.decoder(
            dec,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.head(self.ln_f(out))

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            result["loss"] = loss
        return result
