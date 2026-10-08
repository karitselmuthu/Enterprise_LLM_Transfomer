import json
import tempfile
import unittest
from pathlib import Path

from src.inference.promotion import validate_promotion


class PromotionTests(unittest.TestCase):
    def test_requires_matching_approved_real_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            model = {"version": "v1", "split_sha256": "real-dataset-hash"}
            for partition in ("validation", "test"):
                (root / f"{partition}.json").write_text(json.dumps({
                    "version": "v1", "partition": partition,
                    "dataset_sha256": "real-dataset-hash", "test_count": 12,
                }), encoding="utf-8")
            approval = {
                "status": "pending", "data_type": "real", "model_path": "model.json",
                "model_version": "v1", "dataset_sha256": "real-dataset-hash",
                "validation_report": "validation.json", "test_report": "test.json",
                "approved_by": "reviewer", "approved_at": "2026-10-07",
                "deployment_target": "test-endpoint",
            }
            path = root / "promotion.json"
            path.write_text(json.dumps(approval), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approval"):
                validate_promotion(path, model_path, model)
            approval["status"] = "approved"
            path.write_text(json.dumps(approval), encoding="utf-8")
            self.assertEqual(validate_promotion(path, model_path, model)["status"], "approved")
            approval["dataset_sha256"] = "wrong"
            path.write_text(json.dumps(approval), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset hash"):
                validate_promotion(path, model_path, model)


if __name__ == "__main__":
    unittest.main()
