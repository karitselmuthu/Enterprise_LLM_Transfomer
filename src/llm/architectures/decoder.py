"""Tiny causal decoder for teaching next-token prediction, not production replies."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.llm.attention.variants import CausalAttention, KVCache
from src.llm.context.cache import cache_length
from src.llm.moe.routing import Top1MoE


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    dimension: int = 32
    heads: int = 4
    kv_heads: int = 4
    layers: int = 2
    max_context: int = 96
    experts: int = 0

    def __post_init__(self):
        if self.vocab_size < 4 or self.layers < 1 or self.max_context < 2:
            raise ValueError("Invalid decoder vocabulary, layers, or context length")
        if self.experts not in (0,) and self.experts < 2:
            raise ValueError("Use zero or at least two experts")
        # Validate attention dimensions before constructing the full model.
        CausalAttention(self.dimension, self.heads, self.kv_heads)

    def to_dict(self) -> dict:
        return asdict(self)


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.dimension)
        self.attention = CausalAttention(config.dimension, config.heads, config.kv_heads)
        self.feed_norm = nn.LayerNorm(config.dimension)
        self.feed = (Top1MoE(config.dimension, config.experts) if config.experts else
                     nn.Sequential(nn.Linear(config.dimension, 4 * config.dimension), nn.GELU(),
                                   nn.Linear(4 * config.dimension, config.dimension)))

    def forward(self, x: torch.Tensor, cache: KVCache | None = None):
        attended, new_cache = self.attention(self.attention_norm(x), cache)
        x = x + attended
        feed_result = self.feed(self.feed_norm(x))
        feed_output = feed_result[0] if isinstance(feed_result, tuple) else feed_result
        return x + feed_output, new_cache


class TinyDecoderLM(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dimension)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.dimension)
        self.output = nn.Linear(config.dimension, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, cache: list[KVCache] | None = None):
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.dtype != torch.long:
            raise ValueError("Decoder tokens must be a nonempty int64 [batch, time] tensor")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("KV cache must have one entry per decoder layer")
        if cache_length(cache) + tokens.shape[1] > self.config.max_context:
            raise ValueError("Input exceeds decoder context length")
        x = self.embedding(tokens)
        new_cache = []
        for index, block in enumerate(self.blocks):
            x, layer_cache = block(x, None if cache is None else cache[index])
            new_cache.append(layer_cache)
        return self.output(self.norm(x)), new_cache

    def next_token_loss(self, tokens: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError("Next-token training needs at least two tokens")
        logits, _ = self(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, self.config.vocab_size),
                               tokens[:, 1:].reshape(-1), ignore_index=pad_id)
