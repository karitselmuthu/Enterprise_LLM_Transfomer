"""Evaluate every available generation against the same locked test IDs."""

import argparse
import json
from pathlib import Path

from src.evaluation.evaluate import evaluate
from src.inference.runtime import load_predictor
from src.training.train import load_rows


def compare(paths: list[Path], rows: list[dict[str, str]], partition: str = "test") -> dict:
    results = {}
    reference = None
    for path in paths:
        metadata, predictor = load_predictor(path)
        key = "test_ids" if partition == "test" else "validation_ids"
        identity = (metadata.get("split_sha256"), tuple(metadata.get(key, [])))
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("All models must use the same dataset and test IDs")
        version = metadata["version"]
        if version in results:
            raise ValueError(f"Duplicate generation: {version}")
        results[version] = evaluate(metadata, rows, predictor, partition)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--models", nargs="+", type=Path,
                        default=[Path(f"models/v{i}.json") for i in range(1, 8)])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partition", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    results = compare(args.models, load_rows(args.data), args.partition)
    summary = {
        version: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"],
                  "test_count": value["test_count"], "mistake_ids": [m["id"] for m in value["mistakes"]]}
        for version, value in results.items()
    }
    payload = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
