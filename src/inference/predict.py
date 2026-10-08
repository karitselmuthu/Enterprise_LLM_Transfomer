"""Classify one ticket using a saved V1 or V2 model."""

import argparse
import json
from pathlib import Path

from src.inference.runtime import load_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Ticket text to classify")
    parser.add_argument("--model", type=Path, default=Path("models/v1.json"))
    args = parser.parse_args()
    _, predictor = load_predictor(args.model)
    print(json.dumps(predictor(args.text), indent=2))


if __name__ == "__main__":
    main()
