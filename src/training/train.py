"""Train a V1 model using a locked split manifest."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import random

from src.baselines.traditional import fit


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "text", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain id, text, and label columns")
        rows = [{key: (row[key] or "").strip() for key in ("id", "text", "label")} for row in reader]
    if not rows or any(not all(row.values()) for row in rows):
        raise ValueError("Every row needs a nonempty id, text, and label")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Ticket IDs must be unique")
    return rows


def split_rows(rows: list[dict[str, str]], seed: int = 42, test_fraction: float = 0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    rng = random.Random(seed)
    test_ids = set()
    for label in sorted(groups):
        group = sorted(groups[label], key=lambda row: row["id"])
        if len(group) < 2:
            raise ValueError(f"Label {label!r} needs at least two rows")
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * test_fraction)))
        test_ids.update(row["id"] for row in group[:count])
    train = [row for row in rows if row["id"] not in test_ids]
    test = [row for row in rows if row["id"] in test_ids]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/v1.json"))
    parser.add_argument("--split", type=Path, default=Path("configs/sample_split.json"))
    args = parser.parse_args()

    from src.training.split import load_split

    rows = load_rows(args.data)
    manifest, train, test = load_split(args.split, rows)
    model = fit(train)
    model["split_sha256"] = manifest["dataset_sha256"]
    model["test_ids"] = manifest["test_ids"]
    if "validation_ids" in manifest:
        model["validation_ids"] = manifest["validation_ids"]
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.model}: {len(train)} train, {len(test)} test, {len(model['vocabulary'])} words")


if __name__ == "__main__":
    main()
