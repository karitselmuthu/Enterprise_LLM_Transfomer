"""Persist and verify a ticket split shared by every generation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from src.training.train import load_rows, split_rows


def fingerprint(rows: list[dict[str, str]]) -> str:
    """Hash normalized rows in ID order so row order does not affect the split."""
    canonical = json.dumps(sorted(rows, key=lambda row: row["id"]), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_incident_groups(path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    """Map linked tickets to an incident; unlinked tickets form singleton groups."""
    ticket_ids = {row["id"] for row in rows}
    linked = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"ticket_id", "incident_id"}.issubset(reader.fieldnames):
            raise ValueError("Group CSV must contain ticket_id and incident_id")
        for row in reader:
            ticket_id = (row["ticket_id"] or "").strip()
            incident_id = (row["incident_id"] or "").strip()
            if not ticket_id or not incident_id or ticket_id not in ticket_ids:
                raise ValueError(f"Invalid group link: {ticket_id!r} -> {incident_id!r}")
            if ticket_id in linked:
                raise ValueError(f"Ticket {ticket_id!r} has more than one incident link")
            linked[ticket_id] = f"incident:{incident_id}"
    return {ticket_id: linked.get(ticket_id, f"ticket:{ticket_id}") for ticket_id in ticket_ids}


def split_grouped(rows: list[dict[str, str]], group_ids: dict[str, str], seed: int = 42,
                  test_fraction: float = 0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if set(group_ids) != {row["id"] for row in rows}:
        raise ValueError("Every ticket needs exactly one split group")
    grouped = {}
    for row in rows:
        grouped.setdefault(group_ids[row["id"]], []).append(row)
    by_label = {}
    for group_id, members in grouped.items():
        labels = {row["label"] for row in members}
        if len(labels) != 1:
            raise ValueError(f"Group {group_id!r} spans labels: {sorted(labels)}")
        by_label.setdefault(next(iter(labels)), []).append((group_id, members))

    rng = random.Random(seed)
    chosen_groups = set()
    for label in sorted(by_label):
        candidates = sorted(by_label[label])
        if len(candidates) < 2:
            raise ValueError(f"Label {label!r} needs at least two independent groups")
        rng.shuffle(candidates)
        total = sum(len(members) for _, members in candidates)
        target = total * test_fraction
        # Keep one reproducible subset per attainable size; select the nearest 80/20 split.
        subsets = {0: ()}
        for index, (_, members) in enumerate(candidates):
            additions = {size + len(members): subset + (index,)
                         for size, subset in list(subsets.items())
                         if size + len(members) not in subsets}
            subsets.update(additions)
        possible = [size for size in subsets if 0 < size < total]
        chosen_size = min(possible, key=lambda size: (abs(size - target), size))
        chosen_groups.update(candidates[index][0] for index in subsets[chosen_size])
    test_ids = {row["id"] for row in rows if group_ids[row["id"]] in chosen_groups}
    return ([row for row in rows if row["id"] not in test_ids],
            [row for row in rows if row["id"] in test_ids])


def create_manifest(rows: list[dict[str, str]], seed: int = 42,
                    group_ids: dict[str, str] | None = None,
                    validation_fraction: float = 0.0) -> dict:
    if not 0 <= validation_fraction < 1 or validation_fraction + 0.2 >= 1:
        raise ValueError("validation_fraction must leave training rows after the 20% test split")
    train, test = (split_grouped(rows, group_ids, seed=seed) if group_ids is not None
                   else split_rows(rows, seed=seed))
    validation = []
    if validation_fraction:
        fraction_of_remaining = validation_fraction / 0.8
        if group_ids is not None:
            remaining_groups = {row["id"]: group_ids[row["id"]] for row in train}
            train, validation = split_grouped(train, remaining_groups, seed=seed + 1,
                                              test_fraction=fraction_of_remaining)
        else:
            train, validation = split_rows(train, seed=seed + 1,
                                           test_fraction=fraction_of_remaining)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "dataset_sha256": fingerprint(rows),
        "labels": sorted({row["label"] for row in rows}),
        "train_ids": sorted(row["id"] for row in train),
        "test_ids": sorted(row["id"] for row in test),
    }
    if validation_fraction:
        manifest["validation_ids"] = sorted(row["id"] for row in validation)
        manifest["validation_fraction"] = validation_fraction
    if group_ids is not None:
        manifest["group_ids"] = dict(sorted(group_ids.items()))
        manifest["split_strategy"] = "incident_grouped"
    return manifest


def load_split(path: Path, rows: list[dict[str, str]]) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split manifest version")
    if manifest.get("dataset_sha256") != fingerprint(rows):
        raise ValueError("Dataset differs from the locked split; create a new split explicitly")
    train_ids = set(manifest["train_ids"])
    test_ids = set(manifest["test_ids"])
    all_ids = {row["id"] for row in rows}
    validation_ids = set(manifest.get("validation_ids", []))
    if train_ids & test_ids or train_ids & validation_ids or test_ids & validation_ids:
        raise ValueError("Split partitions must be disjoint")
    if train_ids | test_ids | validation_ids != all_ids:
        raise ValueError("Split IDs must cover the dataset")
    if manifest.get("labels") != sorted({row["label"] for row in rows}):
        raise ValueError("Split labels differ from the dataset")
    if "group_ids" in manifest:
        group_ids = manifest["group_ids"]
        if set(group_ids) != all_ids:
            raise ValueError("Split groups do not cover the dataset")
        group_sides = {}
        for ticket_id, group_id in group_ids.items():
            side = "test" if ticket_id in test_ids else "validation" if ticket_id in validation_ids else "train"
            if group_id in group_sides and group_sides[group_id] != side:
                raise ValueError(f"Group {group_id!r} crosses the split")
            group_sides[group_id] = side
    ordered = sorted(rows, key=lambda row: row["id"])
    return manifest, [row for row in ordered if row["id"] in train_ids], [row for row in ordered if row["id"] in test_ids]


def validation_rows(manifest: dict, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the development partition; keep it out of model fitting."""
    validation_ids = set(manifest.get("validation_ids", []))
    return [row for row in sorted(rows, key=lambda row: row["id"]) if row["id"] in validation_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--output", type=Path, default=Path("configs/sample_split.json"))
    parser.add_argument("--groups", type=Path, help="Optional ticket_id,incident_id CSV")
    parser.add_argument("--validation-fraction", type=float, default=0.0,
                        help="Fraction of all rows reserved for development, in addition to 20%% test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load_rows(args.data)
    group_ids = read_incident_groups(args.groups, rows) if args.groups else None
    manifest = create_manifest(rows, seed=args.seed, group_ids=group_ids,
                               validation_fraction=args.validation_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {len(manifest['train_ids'])} train, "
          f"{len(manifest.get('validation_ids', []))} validation, {len(manifest['test_ids'])} test")


if __name__ == "__main__":
    main()
