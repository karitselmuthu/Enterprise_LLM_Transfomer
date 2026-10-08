"""Compare V1 and V2 on exactly the same locked test tickets."""

import argparse
import json
from pathlib import Path

from src.evaluation.evaluate import evaluate
from src.training.train import load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--v1", type=Path, default=Path("models/v1.json"))
    parser.add_argument("--v2", type=Path, default=Path("models/v2.json"))
    args = parser.parse_args()
    first = json.loads(args.v1.read_text(encoding="utf-8"))
    second = json.loads(args.v2.read_text(encoding="utf-8"))
    if first["version"] != "v1" or second["version"] != "v2":
        raise ValueError("Comparison requires V1 and V2 model files")
    if first.get("split_sha256") != second.get("split_sha256") or first["test_ids"] != second["test_ids"]:
        raise ValueError("Models were not trained against the same split")
    rows = load_rows(args.data)
    results = {"v1": evaluate(first, rows), "v2": evaluate(second, rows)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
