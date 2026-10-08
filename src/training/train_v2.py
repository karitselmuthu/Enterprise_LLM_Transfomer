"""Train V2 Word2Vec on the training IDs in the shared split manifest."""

import argparse
import json
from pathlib import Path

from src.baselines.word2vec import fit
from src.training.split import load_split
from src.training.train import load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/sample_split.json"))
    parser.add_argument("--model", type=Path, default=Path("models/v2.json"))
    parser.add_argument("--dimensions", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    manifest, train, test = load_split(args.split, load_rows(args.data))
    model = fit(train, dimensions=args.dimensions, epochs=args.epochs, seed=manifest["seed"])
    model["split_sha256"] = manifest["dataset_sha256"]
    model["test_ids"] = manifest["test_ids"]
    if "validation_ids" in manifest:
        model["validation_ids"] = manifest["validation_ids"]
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.model}: {len(train)} train, {len(test)} test, {len(model['embedding']['vectors'])} words")


if __name__ == "__main__":
    main()
