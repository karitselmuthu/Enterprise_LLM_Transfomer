"""Causal MHA, MQA, and GQA with an optional KV cache."""

import math

import torch
from torch import nn

from src.llm.positional_encoding.rope import apply_rope


KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, kv_heads: int | None = None):
        super().__init__()
        kv_heads = heads if kv_heads is None else kv_heads
        if dimension < 2 or heads < 1 or kv_heads < 1 or dimension % heads or heads % kv_heads:
            raise ValueError("dimension must divide heads, and heads must divide into kv_heads")
        head_dim = dimension // heads
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, kv_heads * head_dim, bias=False)
        self.value = nn.Linear(dimension, kv_heads * head_dim, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)

    @property
    def variant(self) -> str:
        if self.kv_heads == self.heads:
            return "mha"
        return "mqa" if self.kv_heads == 1 else "gqa"

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> tuple[torch.Tensor, KVCache]:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("attention input must be [batch, time, dimension]")
        batch, length, _ = x.shape
        start = 0 if cache is None else cache[0].shape[-2]
        query = self.query(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rope(query, start), apply_rope(key, start)
        if cache is not None:
            if (cache[0].shape[:2] != (batch, self.kv_heads) or
                    cache[0].shape != cache[1].shape or
                    cache[0].shape[-1] != self.head_dim):
                raise ValueError("KV cache shape does not match attention layer")
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        new_cache = (key, value)
        repeat = self.heads // self.kv_heads
        keys = key.repeat_interleave(repeat, dim=1)
        values = value.repeat_interleave(repeat, dim=1)
        scores = query @ keys.transpose(-1, -2) / math.sqrt(self.head_dim)
        query_positions = torch.arange(start, start + length, device=x.device)[:, None]
        key_positions = torch.arange(key.shape[-2], device=x.device)[None, :]
        scores = scores.masked_fill(key_positions > query_positions, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ values
        merged = attended.transpose(1, 2).contiguous().view(batch, length, self.heads * self.head_dim)
        return self.output(merged), new_cache
