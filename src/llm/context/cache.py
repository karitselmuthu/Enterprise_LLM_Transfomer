"""KV-cache inspection helpers for incremental decoding experiments."""

from src.llm.attention.variants import KVCache


def cache_length(caches: list[KVCache] | None) -> int:
    if not caches:
        return 0
    lengths = {cache[0].shape[-2] for cache in caches}
    if len(lengths) != 1:
        raise ValueError("Decoder layers have inconsistent cache lengths")
    return lengths.pop()
