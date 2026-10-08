"""V5 trainable Transformer ticket encoder and classifier."""

import torch
from torch import nn

from src.transformer.encoder import EncoderBlock
from src.transformer.positional_encoding import PositionalEncoding


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, dimension: int = 32,
                 heads: int = 4, layers: int = 2, max_length: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.position = PositionalEncoding(dimension, max_length)
        self.blocks = nn.ModuleList(EncoderBlock(dimension, heads) for _ in range(layers))
        self.classifier = nn.Linear(dimension, num_labels)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0)
        hidden = self.position(self.embedding(token_ids))
        for block in self.blocks:
            hidden = block(hidden, mask)
        masked = hidden * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.classifier(pooled)
