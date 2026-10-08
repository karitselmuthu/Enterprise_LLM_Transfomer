"""Train V3 LSTM, V4 attention LSTM, or V5 scratch Transformer."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.baselines.lstm import LSTMClassifier
from src.preprocessing.vocabulary import build_vocabulary, encode
from src.training.split import load_split
from src.training.train import load_rows
from src.transformer.model import TransformerClassifier


def make_model(metadata: dict) -> nn.Module:
    version = metadata["version"]
    if version in ("v3", "v4"):
        return LSTMClassifier(
            len(metadata["vocabulary"]), len(metadata["labels"]),
            embedding_dim=metadata["dimension"], hidden_dim=metadata["dimension"],
            attention=version == "v4",
        )
    if version == "v5":
        return TransformerClassifier(
            len(metadata["vocabulary"]), len(metadata["labels"]),
            dimension=metadata["dimension"], heads=metadata["heads"],
            layers=metadata["layers"], max_length=metadata["max_length"],
        )
    raise ValueError(f"Unsupported scratch neural version: {version}")


def train(data: Path, split: Path, output: Path, version: str, epochs: int = 15,
          dimension: int = 32, max_length: int = 32, learning_rate: float = 0.01) -> dict:
    if version not in ("v3", "v4", "v5"):
        raise ValueError("version must be v3, v4, or v5")
    if epochs < 1 or dimension < 1 or max_length < 1 or learning_rate <= 0:
        raise ValueError("Training hyperparameters must be positive")
    if version == "v5" and dimension % 4:
        raise ValueError("V5 dimension must be divisible by 4 heads")
    manifest, train_rows, _ = load_split(split, load_rows(data))
    labels = manifest["labels"]
    vocabulary = build_vocabulary([row["text"] for row in train_rows])
    metadata = {
        "version": version, "labels": labels, "vocabulary": vocabulary,
        "split_sha256": manifest["dataset_sha256"], "test_ids": manifest["test_ids"],
        "dimension": dimension, "max_length": max_length, "heads": 4, "layers": 2,
        "epochs": epochs, "learning_rate": learning_rate, "seed": manifest["seed"],
        "weights_file": output.with_suffix(".pt").name,
    }
    if "validation_ids" in manifest:
        metadata["validation_ids"] = manifest["validation_ids"]
    torch.manual_seed(manifest["seed"])
    torch.set_num_threads(1)
    model = make_model(metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    inputs = torch.tensor([encode(row["text"], vocabulary, max_length) for row in train_rows])
    targets = torch.tensor([labels.index(row["label"]) for row in train_rows])
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output.with_suffix(".pt"))
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model": str(output), "train_count": len(train_rows), "final_training_loss": round(loss.item(), 6)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v3", "v4", "v5"), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/sample_split.json"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dimension", type=int, default=32)
    args = parser.parse_args()
    output = args.model or Path(f"models/{args.version}.json")
    print(json.dumps(train(args.data, args.split, output, args.version,
                           epochs=args.epochs, dimension=args.dimension)))


if __name__ == "__main__":
    main()
