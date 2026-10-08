"""Download a small pretrained encoder once and save a local copy."""

import argparse
from pathlib import Path


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="prajjwal1/bert-tiny")
    parser.add_argument("--revision", default=None, help="Optional pinned Hub commit")
    parser.add_argument("--output", type=Path, default=Path("models/pretrained_base"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id, revision=args.revision)
    tokenizer.save_pretrained(args.output)
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output / "source.txt").write_text(
        f"model_id={args.model_id}\nrevision={args.revision or 'default'}\n", encoding="utf-8"
    )
    print(f"Saved pretrained encoder {args.model_id} to {args.output}")


if __name__ == "__main__":
    main()
