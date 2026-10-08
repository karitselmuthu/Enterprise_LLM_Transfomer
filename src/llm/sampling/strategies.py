"""Greedy, temperature, top-k, and nucleus sampling."""

import torch


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("Expected [batch, vocabulary] logits")
    if temperature < 0:
        raise ValueError("Temperature must be nonnegative")
    if top_k is not None and not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k must be within vocabulary size")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits / temperature
    if top_k is not None:
        cutoff = scores.topk(top_k, dim=-1).values[:, -1, None]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p is not None and top_p < 1:
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.empty_like(scores).scatter_(1, sorted_indices, sorted_scores)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
