"""One residual self-attention and feed-forward encoder block."""

import torch
from torch import nn

from src.transformer.attention import SelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attention = SelfAttention(dimension, heads)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension), nn.GELU(), nn.Linear(4 * dimension, dimension)
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.attention_norm(hidden + self.attention(hidden, mask))
        return self.output_norm(hidden + self.feed_forward(hidden))
