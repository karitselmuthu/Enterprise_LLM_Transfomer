import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.baselines.traditional import fit
from src.evaluation.data_audit import audit
from src.evaluation.evaluate import evaluate
from src.training.split import create_manifest


class RealDataWorkflowTests(unittest.TestCase):
    def test_three_way_evaluation_and_screening(self):
        rows = [
            {"id": f"A{i}", "text": f"password account problem {i}", "label": "access"}
            for i in range(10)
        ] + [
            {"id": f"N{i}", "text": f"vpn connection problem {i}", "label": "network"}
            for i in range(10)
        ]
        rows[0]["text"] = "Contact test.person@example.com about password reset"
        rows[1]["text"] = rows[0]["text"]
        groups = {row["id"]: f"ticket:{row['id']}" for row in rows}
        manifest = create_manifest(rows, group_ids=groups, validation_fraction=0.2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tickets.csv"
            split_path = root / "split.json"
            with data_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "text", "label"])
                writer.writeheader()
                writer.writerows(rows)
            split_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = audit(data_path, split_path)
        self.assertEqual(report["partition_counts"], {"train": 12, "validation": 4, "test": 4})
        self.assertEqual(report["exact_duplicate_id_groups"], [["A0", "A1"]])
        self.assertEqual(report["possible_personal_data_ids"]["email"], ["A0", "A1"])
        self.assertNotIn(rows[0]["text"], json.dumps(report))
        training = [row for row in rows if row["id"] in manifest["train_ids"]]
        model = fit(training)
        model.update({"split_sha256": manifest["dataset_sha256"],
                      "test_ids": manifest["test_ids"],
                      "validation_ids": manifest["validation_ids"]})
        self.assertEqual(evaluate(model, rows, partition="validation")["test_count"], 4)
        self.assertEqual(evaluate(model, rows, partition="test")["test_count"], 4)


if __name__ == "__main__":
    unittest.main()
