"""Screen a ticket CSV and locked split before model evaluation.

This is a reproducible screening report, not a privacy or label-quality sign-off.
It emits ticket IDs and counts, never raw ticket text.
"""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from src.training.split import load_split, read_incident_groups, validation_rows
from src.training.train import load_rows


EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{8,}\d)(?!\w)")


def audit(data: Path, split: Path | None = None, groups: Path | None = None,
          near_threshold: float = 0.85, max_near_rows: int = 1000) -> dict:
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0, 1]")
    rows = load_rows(data)
    report = {
        "data_file": str(data), "row_count": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "screening_only": True,
    }
    normalized = defaultdict(list)
    for row in rows:
        normalized[" ".join(row["text"].casefold().split())].append(row["id"])
    report["exact_duplicate_id_groups"] = [ids for ids in normalized.values() if len(ids) > 1]
    report["possible_personal_data_ids"] = {
        "email": [row["id"] for row in rows if EMAIL.search(row["text"])],
        "ipv4": [row["id"] for row in rows if IPV4.search(row["text"])],
        "phone_like": [row["id"] for row in rows if PHONE.search(row["text"])],
    }
    if groups:
        group_ids = read_incident_groups(groups, rows)
        report["incident_group_count"] = len(set(group_ids.values()))
        by_group = defaultdict(set)
        for row in rows:
            by_group[group_ids[row["id"]]].add(row["label"])
        report["mixed_label_groups"] = sorted(group for group, labels in by_group.items()
                                               if len(labels) > 1)
    if split:
        manifest, train, test = load_split(split, rows)
        if groups and manifest.get("group_ids") != group_ids:
            raise ValueError("Incident mapping differs from the locked split")
        validation = validation_rows(manifest, rows)
        partitions = {"train": train, "validation": validation, "test": test}
        report["dataset_sha256"] = manifest["dataset_sha256"]
        report["partition_counts"] = {name: len(items) for name, items in partitions.items()}
        report["partition_label_counts"] = {
            name: dict(sorted(Counter(row["label"] for row in items).items()))
            for name, items in partitions.items()
        }
        report["near_duplicate_method"] = (
            f"SequenceMatcher text ratio >= {near_threshold}; screening heuristic only"
        )
        if len(rows) > max_near_rows:
            report["near_duplicate_scan"] = f"skipped: {len(rows)} rows exceeds limit {max_near_rows}"
        else:
            matches = []
            for left_name, right_name in (("train", "validation"), ("train", "test"),
                                          ("validation", "test")):
                for left in partitions[left_name]:
                    left_text = left["text"].casefold()
                    for right in partitions[right_name]:
                        right_text = right["text"].casefold()
                        if min(len(left_text), len(right_text)) / max(len(left_text), len(right_text)) < near_threshold:
                            continue
                        ratio = SequenceMatcher(None, left_text, right_text).ratio()
                        if ratio >= near_threshold:
                            matches.append({"left_id": left["id"], "left_partition": left_name,
                                            "right_id": right["id"], "right_partition": right_name,
                                            "ratio": round(ratio, 3)})
            report["near_duplicate_cross_partition_count"] = len(matches)
            report["near_duplicate_cross_partition_examples"] = matches[:30]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    args = parser.parse_args()
    result = audit(args.data, args.split, args.groups, args.near_threshold)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
