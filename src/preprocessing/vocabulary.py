"""Training-only word vocabulary for the scratch neural models."""

from collections import Counter

from src.preprocessing.tokenizer import tokenize

PAD = "<pad>"
UNK = "<unk>"


def build_vocabulary(texts: list[str], min_count: int = 1) -> dict[str, int]:
    counts = Counter(word for text in texts for word in tokenize(text))
    return {word: index for index, word in enumerate([PAD, UNK] + sorted(
        word for word, count in counts.items() if count >= min_count
    ))}


def encode(text: str, vocabulary: dict[str, int], max_length: int) -> list[int]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    ids = [vocabulary.get(word, vocabulary[UNK]) for word in tokenize(text)[:max_length]]
    return (ids or [vocabulary[UNK]]) + [vocabulary[PAD]] * (max_length - max(1, len(ids)))
