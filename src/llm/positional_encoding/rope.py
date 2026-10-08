"""Rotary positions for query/key tensors shaped [batch, heads, time, dim]."""

import torch


def apply_rope(x: torch.Tensor, start_position: int = 0, base: float = 10000.0) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("RoPE expects [batch, heads, time, even head_dim]")
    if start_position < 0 or base <= 0:
        raise ValueError("Invalid RoPE position or base")
    half = x.shape[-1] // 2
    positions = torch.arange(start_position, start_position + x.shape[-2], device=x.device,
                             dtype=torch.float32)
    inverse = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = positions[:, None] * inverse[None, :]
    cosine = angle.cos().to(x.dtype)[None, None, :, :]
    sine = angle.sin().to(x.dtype)[None, None, :, :]
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cosine - right * sine, left * sine + right * cosine), dim=-1)
