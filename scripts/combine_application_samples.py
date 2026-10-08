"""Combine the four application CSVs into one pipeline-ready teaching dataset."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/samples/applications"
SOURCES = {
    "okta_tickets.csv": "Okta",
    "jira_tickets.csv": "Jira",
    "microsoft365_tickets.csv": "Microsoft 365",
    "workday_tickets.csv": "Workday",
}
LABELS = {"access", "hardware", "network", "software"}
COLUMNS = ["id", "text", "label", "application"]


def main() -> None:
    rows = []
    ids = set()
    texts = set()
    for filename, application in SOURCES.items():
        with (DIRECTORY / filename).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != COLUMNS:
                raise ValueError(f"{filename} must have columns {COLUMNS}")
            for row in reader:
                if not all(row.values()) or row["label"] not in LABELS or row["application"] != application:
                    raise ValueError(f"Invalid ticket in {filename}: {row.get('id')}")
                if row["id"] in ids or row["text"].casefold().strip() in texts:
                    raise ValueError(f"Duplicate ticket ID or text: {row['id']}")
                ids.add(row["id"])
                texts.add(row["text"].casefold().strip())
                rows.append(row)
    destination = DIRECTORY / "all_applications.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {destination}: {len(rows)} tickets across {len(SOURCES)} applications")


if __name__ == "__main__":
    main()
