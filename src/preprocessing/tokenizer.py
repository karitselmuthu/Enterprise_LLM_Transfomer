"""Simple, inspectable V1 tokenization."""

import re


def tokenize(text: str) -> list[str]:
    """Return lowercase Unicode word tokens, keeping digits within words."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
