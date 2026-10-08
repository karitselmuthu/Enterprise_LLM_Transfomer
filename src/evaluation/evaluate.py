"""Evaluate V1 or V2 on the held-out ticket IDs saved with the model."""

import argparse
import json
from pathlib import Path

from src.baselines import traditional, word2vec
from src.training.split import fingerprint
from src.training.train import load_rows


def evaluate(model: dict, rows: list[dict[str, str]], predictor=None,
             partition: str = "test") -> dict:
    if "split_sha256" in model and model["split_sha256"] != fingerprint(rows):
        raise ValueError("Dataset differs from the one used to train this model")
    if partition not in ("test", "validation"):
        raise ValueError("partition must be test or validation")
    key = "test_ids" if partition == "test" else "validation_ids"
    if key not in model:
        raise ValueError(f"Model has no {partition} partition")
    target_ids = set(model[key])
    target_rows = [row for row in rows if row["id"] in target_ids]
    if not target_ids or len(target_rows) != len(target_ids):
        raise ValueError(f"Dataset is missing one or more {partition} ticket IDs")
    labels = model["labels"]
    if predictor is None:
        if model["version"] == "v1":
            predictor = lambda text: traditional.predict(model, text)
        elif model["version"] == "v2":
            predictor = lambda text: word2vec.predict(model, text)
        else:
            raise ValueError("Pass a loaded predictor for V3 or later")
    if any(row["label"] not in labels for row in target_rows):
        raise ValueError("Held-out labels differ from training labels")
    predictions = [
        {"id": row["id"], "expected": row["label"], "predicted": predictor(row["text"])["label"]}
        for row in target_rows
    ]
    metrics = {}
    for label in labels:
        tp = sum(row["expected"] == label and row["predicted"] == label for row in predictions)
        fp = sum(row["expected"] != label and row["predicted"] == label for row in predictions)
        fn = sum(row["expected"] == label and row["predicted"] != label for row in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    return {
        "version": model["version"],
        "partition": partition,
        "dataset_sha256": model.get("split_sha256"),
        "test_count": len(predictions),
        "accuracy": sum(row["expected"] == row["predicted"] for row in predictions) / len(predictions),
        "macro_f1": sum(metrics[label]["f1"] for label in labels) / len(labels),
        "per_class": metrics,
        "mistakes": [row for row in predictions if row["expected"] != row["predicted"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/v1.json"))
    parser.add_argument("--partition", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    from src.inference.runtime import load_predictor
    model, predictor = load_predictor(args.model)
    print(json.dumps(evaluate(model, load_rows(args.data), predictor, args.partition), indent=2))


if __name__ == "__main__":
    main()
