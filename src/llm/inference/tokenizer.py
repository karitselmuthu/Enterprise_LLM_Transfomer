"""Small word tokenizer for the synthetic decoder demonstration."""

import re


SPECIAL = ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")
TOKEN = re.compile(r"[a-z0-9]+|[^\w\s]", re.I)


class DemoTokenizer:
    def __init__(self, vocabulary: dict[str, int]):
        if any(vocabulary.get(token) != index for index, token in enumerate(SPECIAL)):
            raise ValueError("Vocabulary must start with ordered special tokens")
        self.vocabulary = vocabulary
        self.reverse = {index: token for token, index in vocabulary.items()}

    @classmethod
    def build(cls, texts: list[str]):
        words = sorted({token for text in texts for token in TOKEN.findall(text.lower())})
        vocabulary = {token: index for index, token in enumerate(SPECIAL)}
        vocabulary.update({word: len(vocabulary) + index for index, word in enumerate(
            word for word in words if word not in vocabulary)})
        return cls(vocabulary)

    def encode(self, text: str) -> list[int]:
        return [self.vocabulary.get(token, self.vocabulary["<unk>"])
                for token in TOKEN.findall(text.lower())]

    def decode(self, ids: list[int]) -> str:
        words = [self.reverse.get(index, "<unk>") for index in ids
                 if self.reverse.get(index) not in SPECIAL]
        result = " ".join(words)
        return re.sub(r"\s+([.,!?;:])", r"\1", result).strip()
