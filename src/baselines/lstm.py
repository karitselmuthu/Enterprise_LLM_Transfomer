"""V3 LSTM and V4 LSTM plus attention classifiers."""

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embedding_dim: int = 32,
                 hidden_dim: int = 32, attention: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1) if attention else None
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0)
        embedded = self.embedding(token_ids)
        outputs, _ = self.lstm(embedded)
        if self.attention is None:
            last_index = mask.sum(dim=1).sub(1).clamp(min=0)
            pooled = outputs[torch.arange(outputs.size(0), device=outputs.device), last_index]
        else:
            scores = self.attention(outputs).squeeze(-1).masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)
