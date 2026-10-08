"""Small top-1 mixture of experts; overflow keeps the residual token."""

import math

import torch
from torch import nn


class Top1MoE(nn.Module):
    def __init__(self, dimension: int, experts: int = 4, hidden: int | None = None,
                 capacity_factor: float = 1.25):
        super().__init__()
        if dimension < 1 or experts < 2 or capacity_factor <= 0:
            raise ValueError("Invalid MoE dimension, expert count, or capacity")
        self.router = nn.Linear(dimension, experts)
        hidden = hidden or 4 * dimension
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                                   nn.Linear(hidden, dimension))
                                     for _ in range(experts))
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if x.ndim != 3:
            raise ValueError("MoE input must be [batch, time, dimension]")
        flat = x.reshape(-1, x.shape[-1])
        probabilities = torch.softmax(self.router(flat), dim=-1)
        choices = probabilities.argmax(dim=-1)
        capacity = max(1, math.ceil(self.capacity_factor * len(flat) / len(self.experts)))
        output = flat.clone()
        counts = []
        overflow = 0
        for index, expert in enumerate(self.experts):
            selected = torch.nonzero(choices == index, as_tuple=True)[0]
            selected = selected[torch.argsort(probabilities[selected, index], descending=True)]
            counts.append(int(len(selected)))
            overflow += max(0, len(selected) - capacity)
            accepted = selected[:capacity]
            if len(accepted):
                output[accepted] = expert(flat[accepted]) * probabilities[accepted, index, None]
        return output.view_as(x), {"counts": counts, "overflow": overflow, "capacity": capacity}
