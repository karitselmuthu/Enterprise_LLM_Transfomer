"""Sinusoidal position vectors added to token embeddings."""

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) *
                           (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        return embedded + self.encoding[:, :embedded.size(1)]
