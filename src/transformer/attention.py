"""Explicit Q/K/V scaled dot-product multi-head self-attention."""

import math
import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.heads, self.head_dimension).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self._split(self.query(hidden))
        k = self._split(self.key(hidden))
        v = self._split(self.value(hidden))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dimension)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        batch, _, length, _ = context.shape
        merged = context.transpose(1, 2).reshape(batch, length, self.heads * self.head_dimension)
        return self.output(merged)
