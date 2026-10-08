"""Show nearest words in a trained V2 embedding."""

import argparse
import json
from pathlib import Path

from src.embeddings.word2vec import nearest_words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word", help="A word present in V2's training vocabulary")
    parser.add_argument("--model", type=Path, default=Path("models/v2.json"))
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    if model.get("version") != "v2":
        raise ValueError("Embedding inspection requires a V2 model")
    for word, similarity in nearest_words(model["embedding"], args.word.lower(), args.count):
        print(f"{word}\t{similarity:.4f}")


if __name__ == "__main__":
    main()
