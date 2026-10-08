"""Train V6 frozen-encoder centroids or fine-tune a V7 classifier."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.training.split import load_split
from src.training.train import load_rows


def _encode(tokenizer, texts: list[str], max_length: int) -> dict:
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def train(data: Path, split: Path, output: Path, base_dir: Path, version: str,
          epochs: int = 8, max_length: int = 64) -> dict:
    if version not in ("v6", "v7"):
        raise ValueError("version must be v6 or v7")
    if epochs < 1 or max_length < 1:
        raise ValueError("epochs and max_length must be positive")
    if not (base_dir / "config.json").exists():
        raise FileNotFoundError("Prepare local pretrained weights first")
    manifest, train_rows, _ = load_split(split, load_rows(data))
    torch.manual_seed(manifest["seed"])
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    batch = _encode(tokenizer, [row["text"] for row in train_rows], max_length)
    labels = manifest["labels"]
    metadata = {
        "version": version, "labels": labels,
        "split_sha256": manifest["dataset_sha256"], "test_ids": manifest["test_ids"],
        "max_length": max_length,
    }
    if "validation_ids" in manifest:
        metadata["validation_ids"] = manifest["validation_ids"]
    if version == "v6":
        encoder = AutoModel.from_pretrained(base_dir, local_files_only=True)
        encoder.eval()
        with torch.inference_mode():
            features = encoder(**batch).last_hidden_state[:, 0, :]
            features = nn.functional.normalize(features, dim=1)
        centroids = {}
        for label in labels:
            indices = [i for i, row in enumerate(train_rows) if row["label"] == label]
            centroid = features[indices].mean(dim=0)
            centroids[label] = nn.functional.normalize(centroid, dim=0).tolist()
        metadata.update({"base_dir": os.path.relpath(base_dir.resolve(), output.parent.resolve()),
                         "centroids": centroids})
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_dir, num_labels=len(labels), local_files_only=True,
            id2label={i: label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )
        targets = torch.tensor([labels.index(row["label"]) for row in train_rows])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = model(**batch, labels=targets).loss
            loss.backward()
            optimizer.step()
        model_dir = output.with_suffix("")
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        metadata.update({"model_dir": os.path.relpath(model_dir.resolve(), output.parent.resolve()),
                         "epochs": epochs})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model": str(output), "train_count": len(train_rows), "version": version}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v6", "v7"), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--split", type=Path, default=Path("configs/sample_split.json"))
    parser.add_argument("--base-dir", type=Path, default=Path("models/pretrained_base"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    output = args.model or Path(f"models/{args.version}.json")
    print(json.dumps(train(args.data, args.split, output, args.base_dir, args.version, epochs=args.epochs)))


if __name__ == "__main__":
    main()
