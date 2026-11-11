"""
model.py — MongolHistorySLM
===========================

A decoder-only language model for Mongolian history, built on the modern small-model
recipe used by Llama 3, Qwen 3 and SmolLM2. It is one standalone file with no dependencies
besides PyTorch (>= 2.1). The size is a preset: from "nano" (about 1M parameters, trains on
a CPU) to "base" (about 104M, for large public corpora). See ModelConfig.preset.

Architecture
    * Decoder-only Transformer, pre-normalization (x + f(norm(x)))
    * RMSNorm everywhere (gain only, no bias), always computed in fp32
    * Rotary Position Embeddings (RoPE), "half-split" layout as in HF Llama/Qwen
    * Grouped-Query Attention: several query heads share one K/V head (9 and 3 in the base
      preset), so the KV cache is 3x smaller
    * QK-Norm (Qwen3 / OLMo-2): RMSNorm on each head's q and k, which keeps fp16 training stable
    * SwiGLU feed-forward network
    * Input embedding tied to the LM head, and no bias terms anywhere
    * "Deep and thin" shape (24 x 576 in the base preset): below ~1B params, depth beats
      width (MobileLLM, Liu et al. 2024; SmolLM2 uses the same 576-wide block)
    * Optional weight sharing across depth (n_unique_layers): neighbouring layers reuse
      one block. Same compute, fewer parameters, and in the README's experiment a slightly
      higher loss, so it is off by default.
    * With sharing off, mathematically identical to Hugging Face `Qwen3ForCausalLM`
      (verified: logits match to 1e-6 once the fused qkv / gate_up matrices are split), so
      trained weights can be exported later to TRL, vLLM or llama.cpp

Efficiency on free GPUs (Colab / Kaggle T4, P100, A100)
    * bf16 autocast on A100/L4, fp16 autocast + GradScaler on T4/P100 (see `autocast_dtype`)
    * Fused QKV and gate/up projections, so each sublayer does one large matmul
    * Chunked LM head + cross-entropy: the (B, T, 32768) logits tensor is never stored
    * Optional activation checkpointing (`model.gradient_checkpointing = True`)
    * PyTorch SDPA kernels (FlashAttention / memory-efficient) and a KV cache for generation

Minimal training step:

    dtype  = autocast_dtype()                                  # bf16 on A100, fp16 on T4/P100
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    with torch.autocast("cuda", dtype=dtype):                  # == torch.cuda.amp.autocast(dtype=...)
        _, loss = model(x, y)                                  # y = x shifted left by one token
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

Run `python model.py` to print the parameter budget and run a quick self-test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100  # label value excluded from the loss (prompt tokens, padding)


# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # The tokenizer (trained later on Mongolian + English text) must have <= vocab_size
    # tokens. 32768 is a multiple of 128, which keeps the LM-head matmul tensor-core friendly.
    vocab_size: int = 32_768
    d_model: int = 576           # residual stream width
    n_layers: int = 24           # depth: how many blocks the residual stream passes through
    n_unique_layers: Optional[int] = None   # distinct blocks; None = n_layers (no sharing).
                                 # With fewer, neighbouring layers share one set of weights:
                                 # 6 layers on 3 unique blocks run A A B B C C.
    n_heads: int = 9             # query heads
    n_kv_heads: int = 3          # key/value heads (GQA: each K/V head serves 3 query heads)
    head_dim: int = 64           # n_heads * head_dim == d_model
    ffn_hidden: int = 1536       # SwiGLU inner width, 8/3 * d_model
    max_seq_len: int = 2048      # context window
    rope_theta: float = 100_000.0  # RoPE base; >10k leaves room for later context extension
    norm_eps: float = 1e-5
    qk_norm: bool = True
    tie_embeddings: bool = True
    dropout: float = 0.0         # 0 for pretraining; ~0.1 can help small-data fine-tuning
    init_std: float = 0.02
    ce_chunk_size: int = 2048    # tokens per LM-head/CE chunk (bounds the logits memory)

    def __post_init__(self) -> None:
        if self.n_unique_layers is None:
            self.n_unique_layers = self.n_layers
        if not 1 <= self.n_unique_layers <= self.n_layers:
            raise ValueError("n_unique_layers must be between 1 and n_layers")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @classmethod
    def preset(cls, name: str, **overrides) -> "ModelConfig":
        """A named size. Pass vocab_size (your tokenizer's) and anything else to override:
        ModelConfig.preset("micro", vocab_size=tok.vocab_size, dropout=0.1)."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})


# Sizes. ffn_hidden is 8/3 * d_model, the SwiGLU convention. The parameter counts are for
# the blocks only; the embedding adds vocab_size * d_model.
PRESETS = {
    "nano":  dict(d_model=128, n_layers=6,  n_heads=4, n_kv_heads=2, head_dim=32, ffn_hidden=344,  max_seq_len=512),   # 1.1M
    "micro": dict(d_model=256, n_layers=8,  n_heads=4, n_kv_heads=2, head_dim=64, ffn_hidden=688,  max_seq_len=1024),  # 5.8M
    "small": dict(d_model=384, n_layers=12, n_heads=6, n_kv_heads=2, head_dim=64, ffn_hidden=1024, max_seq_len=2048),  # 19M
    "base":  dict(d_model=576, n_layers=24, n_heads=9, n_kv_heads=3, head_dim=64, ffn_hidden=1536, max_seq_len=2048),  # 85M
}


# ----------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square normalization (Zhang & Sennrich, 2019): no centering, no bias.

    Computed in fp32 and cast back to the input dtype, so it is safe under fp16/bf16.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables (Su et al., 2021).

    The tables are built in float64 and stored as fp32. They are non-persistent buffers,
    so they follow `.to(device)` but are not written into checkpoints.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 100_000.0):
        super().__init__()
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64, device="cpu") / head_dim
        inv_freq = 1.0 / (theta ** exponent)                                  # (head_dim/2,)
        positions = torch.arange(max_seq_len, dtype=torch.float64, device="cpu")
        angles = torch.outer(positions, inv_freq)                             # (T, head_dim/2)
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    def forward(self, start_pos: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + seq_len
        return self.cos[start_pos:end], self.sin[start_pos:end]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs (i, i + D/2) by position-dependent angles.

    x: (B, T, H, D); cos, sin: (T, D/2). The math runs in fp32 and the result is cast back.
    This equals HF's `x * cos + rotate_half(x) * sin`, so exporting to HF needs no weight permutation.
    """
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    It stores only the n_kv_heads K/V heads, which is where GQA saves memory. The buffers
    are allocated on first use, so they take the device and dtype of the keys
    (e.g. fp16 under autocast).
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # k, v: (B, n_kv_heads, T, head_dim)
        if self.k is None:
            b, h, _, d = k.shape
            self.k = k.new_zeros(b, h, self.max_seq_len, d)
            self.v = v.new_zeros(b, h, self.max_seq_len, d)
        end = start_pos + k.size(2)
        if end > self.max_seq_len:
            raise ValueError(f"KV cache overflow: position {end} > {self.max_seq_len}")
        self.k[:, :, start_pos:end] = k
        self.v[:, :, start_pos:end] = v
        return self.k[:, :, :end], self.v[:, :, :end]


class GroupedQueryAttention(nn.Module):
    """Causal multi-head attention where groups of query heads share one K/V head
    (Ainslie et al., 2023), with optional QK-Norm and RoPE."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_size = cfg.n_heads * cfg.head_dim
        self.kv_size = cfg.n_kv_heads * cfg.head_dim
        self.dropout = cfg.dropout

        # One fused projection for q, k, v: a single matmul and a single autocast cast.
        self.qkv_proj = nn.Linear(cfg.d_model, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)              # QK-Norm before RoPE (Qwen3 order)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # (B, heads, T, D)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)        # (B, n_kv_heads, start_pos+T, D)

        # Causal masking. Training and prefill use the fused causal kernel. A single decode
        # step attends to the whole cache. A multi-token chunk appended to an existing cache
        # needs an explicit mask, because `is_causal` aligns the mask to the top-left corner.
        if kv_cache is None or start_pos == 0:
            attn_mask, is_causal = None, T > 1
        elif T == 1:
            attn_mask, is_causal = None, False
        else:
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            k_pos = torch.arange(k.size(2), device=x.device)[None, :]
            attn_mask, is_causal = k_pos <= q_pos, False

        # Expand the K/V heads to match the query heads. The cache stays at n_kv_heads; this
        # copy exists only during the attention call, and it works with every SDPA backend
        # (including the memory-efficient kernel used on T4/P100).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).reshape(B, T, self.q_size)
        return self.resid_dropout(self.o_proj(y))


class SwiGLU(nn.Module):
    """Gated feed-forward network (Shazeer, 2020): down( silu(x W_gate) * (x W_up) )."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * hidden, bias=False)   # [gate | up], fused
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: x + Attn(RMSNorm(x)), then x + SwiGLU(RMSNorm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, kv_cache, start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ----------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------
class MongolHistorySLM(nn.Module):
    """Decoder-only small language model specialised for Mongolian history."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        # `layers` holds the distinct blocks; `layer_map[i]` says which one runs at depth i.
        # Without sharing that is 0, 1, 2, ...; with 6 layers on 3 blocks it is 0 0 1 1 2 2.
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_unique_layers))
        self.layer_map = [i * config.n_unique_layers // config.n_layers for i in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.gradient_checkpointing = False   # set True to trade ~30% compute for activation memory
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """N(0, 0.02) for every weight matrix. The residual-branch output projections
        (o_proj, down_proj) are scaled down by 1/sqrt(2 * n_layers) (GPT-2 / Megatron),
        so the residual stream stays well-conditioned at depth. n_layers is the depth, not
        the number of distinct blocks: a shared block adds to the stream every time it runs."""
        std = self.config.init_std
        resid_std = std / math.sqrt(2 * self.config.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_resid_out = name.endswith(("o_proj", "down_proj"))
                nn.init.normal_(module.weight, mean=0.0, std=resid_std if is_resid_out else std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    # -- forward ---------------------------------------------------------------------------
    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[KVCache]] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Token ids (B, T) -> final normalized hidden states (B, T, d_model).

        kv_caches holds one cache per depth (n_layers of them), also when blocks are shared:
        a shared block sees different inputs at each depth, so its keys and values differ."""
        _, T = input_ids.shape
        if start_pos + T > self.config.max_seq_len:
            raise ValueError(f"sequence end {start_pos + T} exceeds max_seq_len {self.config.max_seq_len}")
        x = self.embed_dropout(self.embed_tokens(input_ids))
        cos, sin = self.rope(start_pos, T)
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for depth, block in enumerate(self.layer_map):
            layer = self.layers[block]
            if use_ckpt:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin, None if kv_caches is None else kv_caches[depth], start_pos)
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        reduction: str = "mean",
        kv_caches: Optional[List[KVCache]] = None,
        start_pos: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token ids.
            labels:    (B, T) next-token targets, already shifted: labels[:, t] is the token
                       that follows input_ids[:, t]. For packed pretraining use
                       x = buf[:-1], y = buf[1:]. Positions set to IGNORE_INDEX (-100) are
                       skipped, e.g. system/user turns during SFT.
            reduction: "mean" (over non-ignored tokens), "sum", or "none" for a per-token
                       NLL of shape (B, T). The "none" output gives sequence log-probs
                       for DPO/ORPO.
            kv_caches: one KVCache per layer, for incremental decoding.
            start_pos: absolute position of input_ids[:, 0] (used with kv_caches).

        Returns:
            (logits, None) when labels is None, with logits of shape (B, T, vocab_size).
            (None, loss) when labels is given. The full logits are never materialized then.
        """
        h = self.forward_hidden(input_ids, kv_caches=kv_caches, start_pos=start_pos)
        if labels is None:
            return self.lm_head(h), None
        return None, self._chunked_lm_loss(h, labels, reduction)

    def _token_nll(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(h).float()   # fp32 log-softmax for a stable loss
        return F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX, reduction="none")

    def _chunked_lm_loss(self, h: torch.Tensor, labels: torch.Tensor, reduction: str) -> torch.Tensor:
        """LM head + cross-entropy over chunks of `ce_chunk_size` tokens.

        During training each chunk is checkpointed: backward stores only the hidden-state
        chunk and recomputes that chunk's logits. A dense loss would keep several fp16/fp32
        copies of the B*T*V logits (about 6 GB at B=8, T=2048); chunking keeps one
        ce_chunk_size*V slice at a time (under 1 GB), for roughly 6% extra compute.
        A batch that fits in one chunk is not checkpointed: there is nothing to save.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")
        B, T, D = h.shape
        h, y = h.reshape(B * T, D), labels.reshape(B * T)
        step = self.config.ce_chunk_size
        recompute = torch.is_grad_enabled() and h.requires_grad and B * T > step
        chunks = []
        for i in range(0, B * T, step):
            hc, yc = h[i:i + step], y[i:i + step]
            if recompute:
                chunks.append(checkpoint(self._token_nll, hc, yc, use_reentrant=False))
            else:
                chunks.append(self._token_nll(hc, yc))
        nll = torch.cat(chunks)   # 0 at ignored positions
        if reduction == "none":
            return nll.view(B, T)
        if reduction == "sum":
            return nll.sum()
        return nll.sum() / (y != IGNORE_INDEX).sum().clamp(min=1)

    # -- generation ------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        logit_hook: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with a KV cache. Set temperature=0 for greedy decoding.

        The prompts in a batch must have equal length (no left padding). Wrap the call in
        torch.autocast for fp16/bf16 inference on GPU.

        logit_hook(h, logits) -> scores, if given, replaces the logits before sampling.
        h is the final hidden state (B, d_model) and logits is (B, vocab_size). The memory
        in memory.py uses it to mix in what it retrieved.
        """
        B, T0 = input_ids.shape
        max_new_tokens = min(max_new_tokens, self.config.max_seq_len - T0)
        if max_new_tokens <= 0:
            raise ValueError("the prompt already fills the context window")
        was_training = self.training
        self.eval()
        try:
            caches = [KVCache(T0 + max_new_tokens) for _ in self.layer_map]
            h = self.forward_hidden(input_ids, kv_caches=caches, start_pos=0)   # prefill
            seq = input_ids
            finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
            for i in range(max_new_tokens):
                logits = self.lm_head(h[:, -1, :]).float()
                if logit_hook is not None:
                    logits = logit_hook(h[:, -1, :].float(), logits)
                next_tok = sample_next_token(logits, seq, temperature, top_k, top_p, repetition_penalty)
                if eos_token_id is not None:
                    next_tok = next_tok.masked_fill(finished, eos_token_id)    # pad finished rows
                    finished |= next_tok == eos_token_id
                seq = torch.cat([seq, next_tok[:, None]], dim=1)
                if i == max_new_tokens - 1 or (eos_token_id is not None and bool(finished.all())):
                    break
                h = self.forward_hidden(next_tok[:, None], kv_caches=caches, start_pos=T0 + i)
            return seq
        finally:
            self.train(was_training)

    # -- utilities -------------------------------------------------------------------------
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Unique trainable parameters; a tied embedding/LM head is counted once."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed_tokens.weight.numel() if exclude_embeddings else n

    def flops_per_token(self, seq_len: Optional[int] = None) -> float:
        """Training FLOPs per token (forward + backward), PaLM convention:
        6 * (matmul params, LM head included) + 12 * n_layers * n_heads * head_dim * seq_len."""
        c = self.config
        T = seq_len or c.max_seq_len
        block = sum(m.weight.numel() for m in self.layers[0].modules() if isinstance(m, nn.Linear))
        matmul_params = block * c.n_layers + self.lm_head.weight.numel()   # a shared block runs at every depth
        return 6 * matmul_params + 12 * c.n_layers * c.n_heads * c.head_dim * T

    def configure_optimizer(
        self,
        lr: float = 2e-3,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matrices only (no decay on RMSNorm gains).
        It uses the fused CUDA kernel when the parameters are on a GPU."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = all(p.is_cuda for p in params)
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def sample_next_token(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """logits: (B, V) fp32; prev_tokens: (B, L), the tokens seen so far. Returns (B,)."""
    if repetition_penalty != 1.0:   # CTRL-style penalty on tokens that already appeared
        seen = logits.gather(1, prev_tokens)
        seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
        logits = logits.scatter(1, prev_tokens, seen)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p   # always keeps the most likely token
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)


def autocast_dtype() -> torch.dtype:
    """Picks the autocast dtype for the current GPU.

    bf16 on Ampere and newer (A100, L4, H100). fp16 on T4, P100 and V100, which have no
    native bf16; pair it with torch.amp.GradScaler.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


def parameter_budget(model: MongolHistorySLM) -> str:
    """Human-readable parameter breakdown."""
    groups = {"token embedding (tied LM head)": 0, "attention (qkv + o)": 0,
              "SwiGLU FFN (gate/up + down)": 0, "RMSNorm gains (incl. QK-Norm)": 0}
    for name, p in model.named_parameters():
        if name.startswith("embed_tokens"):
            groups["token embedding (tied LM head)"] += p.numel()
        elif ".attn." in name and "_norm" not in name:
            groups["attention (qkv + o)"] += p.numel()
        elif ".ffn." in name:
            groups["SwiGLU FFN (gate/up + down)"] += p.numel()
        else:
            groups["RMSNorm gains (incl. QK-Norm)"] += p.numel()
    total = model.num_parameters()
    lines = [f"{'component':<32}{'params':>14}{'share':>9}"]
    lines += [f"{k:<32}{v:>14,}{v / total:>9.1%}" for k, v in groups.items()]
    lines.append(f"{'TOTAL':<32}{total:>14,}{1:>9.1%}")
    lines.append(f"{'non-embedding':<32}{model.num_parameters(exclude_embeddings=True):>14,}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Self-test:  python model.py
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig()
    model = MongolHistorySLM(cfg)
    print(cfg, "\n")
    print(parameter_budget(model), "\n")
    print(f"training FLOPs/token @ T={cfg.max_seq_len}: {model.flops_per_token() / 1e9:.2f} GFLOP")
    kv_bytes = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 2
    print(f"fp16 KV cache: {kv_bytes / 1024:.0f} KiB/token, "
          f"{kv_bytes * cfg.max_seq_len / 2**20:.0f} MiB per full {cfg.max_seq_len}-token sequence\n")

    # 1) Incremental decoding through the KV cache must reproduce the full forward pass.
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        full_logits, _ = model(x)
        caches = [KVCache(64) for _ in model.layer_map]
        parts = [model.lm_head(model.forward_hidden(x[:, :40], caches, 0)),      # prefill
                 model.lm_head(model.forward_hidden(x[:, 40:48], caches, 40))]   # chunked prefill
        for t in range(48, 64):                                                  # token by token
            parts.append(model.lm_head(model.forward_hidden(x[:, t:t + 1], caches, t)))
        inc_logits = torch.cat(parts, dim=1)
    diff = (full_logits - inc_logits).abs().max().item()
    print(f"[check] KV-cache vs full forward, max |dlogit| = {diff:.2e}")
    assert diff < 1e-3

    # 2) The chunked loss matches a dense cross-entropy, and backward runs.
    model.train()
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = IGNORE_INDEX
    _, loss = model(x, y)
    dense = F.cross_entropy(full_logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE_INDEX)
    loss.backward()
    print(f"[check] loss = {loss.item():.4f} (dense {dense.item():.4f}; ln V = {math.log(cfg.vocab_size):.4f})")
    assert abs(loss.item() - dense.item()) < 1e-4

    # 3) Generation runs end to end.
    out = model.generate(x[:1, :8], max_new_tokens=8, temperature=0.8)
    print(f"[check] generate: {tuple(x[:1, :8].shape)} -> {tuple(out.shape)}")
    del model

    # 4) Weight sharing. A shared model must equal an unshared model whose layers hold copies
    #    of the shared weights, and its KV cache must still reproduce the full forward pass.
    small = dict(vocab_size=512, d_model=64, n_layers=6, n_heads=4, n_kv_heads=2, head_dim=16,
                 ffn_hidden=96, max_seq_len=64)
    shared = MongolHistorySLM(ModelConfig(**small, n_unique_layers=3)).eval()
    assert shared.layer_map == [0, 0, 1, 1, 2, 2]
    plain = MongolHistorySLM(ModelConfig(**small)).eval()
    state = {k: v for k, v in shared.state_dict().items() if not k.startswith("layers.")}
    for depth, block in enumerate(shared.layer_map):
        for k, v in shared.layers[block].state_dict().items():
            state[f"layers.{depth}.{k}"] = v
    plain.load_state_dict(state)
    xs = torch.randint(0, 512, (2, 48))
    with torch.no_grad():
        a, _ = shared(xs)
        b, _ = plain(xs)
        caches = [KVCache(48) for _ in shared.layer_map]
        inc = torch.cat([shared.lm_head(shared.forward_hidden(xs[:, :20], caches, 0))] +
                        [shared.lm_head(shared.forward_hidden(xs[:, t:t + 1], caches, t)) for t in range(20, 48)], dim=1)
    n_shared, n_plain = shared.num_parameters(exclude_embeddings=True), plain.num_parameters(exclude_embeddings=True)
    print(f"[check] 6 layers on 3 shared blocks == 6 unshared layers with copied weights, "
          f"max |dlogit| = {(a - b).abs().max().item():.2e}; block params {n_plain:,} -> {n_shared:,}")
    assert (a - b).abs().max().item() < 1e-5 and (a - inc).abs().max().item() < 1e-4
    assert n_plain - 64 == 2 * (n_shared - 64)          # 64 = the final RMSNorm, which is not a block

    # 5) Causality. Changing the last token must change its own output and nothing earlier.
    xs2 = xs.clone()
    xs2[:, -1] = (xs2[:, -1] + 1) % 512
    with torch.no_grad():
        c1, c2 = shared(xs)[0], shared(xs2)[0]
    leak = (c1[:, :-1] - c2[:, :-1]).abs().max().item()
    own = (c1[:, -1] - c2[:, -1]).abs().max().item()
    print(f"[check] changing the last token moves earlier logits by {leak:.1e} and its own by {own:.1e}")
    assert leak < 1e-6 and own > 1e-4

    # 6) The size presets.
    print("\npreset    blocks   total at vocab 4,096")
    for name in PRESETS:
        m = MongolHistorySLM(ModelConfig.preset(name, vocab_size=4096))
        print(f"  {name:<6} {m.num_parameters(exclude_embeddings=True) / 1e6:>6.1f}M  {m.num_parameters() / 1e6:>6.1f}M")
    print("self-test passed")
