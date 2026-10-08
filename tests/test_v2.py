import json
import tempfile
import unittest
from pathlib import Path

from src.baselines import word2vec
from src.embeddings.word2vec import document_vector, nearest_words, train_embeddings
from src.evaluation.evaluate import evaluate
from src.training.split import create_manifest, load_split, split_grouped, validation_rows


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "A1", "text": "password login access", "label": "access"},
            {"id": "A2", "text": "login password reset", "label": "access"},
            {"id": "N1", "text": "vpn network connection", "label": "network"},
            {"id": "N2", "text": "network vpn disconnect", "label": "network"},
        ]

    def test_locked_split_rejects_changed_ticket_text(self):
        manifest = create_manifest(self.rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _, train, test = load_split(path, self.rows)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(test), 2)
            changed = [row.copy() for row in self.rows]
            changed[0]["text"] = "different text"
            with self.assertRaisesRegex(ValueError, "Dataset differs"):
                load_split(path, changed)

    def test_word2vec_is_deterministic_and_handles_unknown_words(self):
        texts = [row["text"] for row in self.rows]
        first = train_embeddings(texts, dimensions=4, epochs=3, seed=7)
        second = train_embeddings(texts, dimensions=4, epochs=3, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(document_vector("unseen", first["vectors"], 4), [0.0] * 4)
        self.assertEqual(len(nearest_words(first, "vpn", count=2)), 2)

    def test_incident_group_never_crosses_split(self):
        rows = self.rows + [
            {"id": "A3", "text": "same incident", "label": "access"},
            {"id": "N3", "text": "another incident", "label": "network"},
        ]
        groups = {row["id"]: f"ticket:{row['id']}" for row in rows}
        groups["A1"] = groups["A2"] = "incident:one"
        train, test = split_grouped(rows, groups)
        train_ids = {row["id"] for row in train}
        test_ids = {row["id"] for row in test}
        self.assertEqual("A1" in train_ids, "A2" in train_ids)
        self.assertFalse(train_ids & test_ids)
        manifest = create_manifest(rows, group_ids=groups)
        self.assertEqual(set(manifest["test_ids"]), test_ids)
        self.assertEqual(len({row["label"] for row in test}), 2)

    def test_mixed_label_incident_is_rejected(self):
        groups = {row["id"]: "incident:mixed" for row in self.rows}
        with self.assertRaisesRegex(ValueError, "spans labels"):
            split_grouped(self.rows, groups)

    def test_three_way_split_excludes_validation_from_training(self):
        rows = [
            {"id": f"{label}{index}", "text": f"{label} issue {index}", "label": label}
            for label in ("access", "network") for index in range(10)
        ]
        groups = {row["id"]: f"ticket:{row['id']}" for row in rows}
        manifest = create_manifest(rows, group_ids=groups, validation_fraction=0.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, train, test = load_split(path, rows)
        validation = validation_rows(loaded, rows)
        self.assertEqual((len(train), len(validation), len(test)), (12, 4, 4))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in validation})
        self.assertEqual({row["label"] for row in validation}, {"access", "network"})

    def test_v2_predicts_and_evaluates_on_saved_ids(self):
        manifest = create_manifest(self.rows)
        training = [row for row in self.rows if row["id"] in manifest["train_ids"]]
        model = word2vec.fit(training, dimensions=4, epochs=3)
        model["split_sha256"] = manifest["dataset_sha256"]
        model["test_ids"] = manifest["test_ids"]
        self.assertIn(word2vec.predict(model, "vpn")["label"], model["labels"])
        result = evaluate(model, self.rows)
        self.assertEqual(result["test_count"], 2)
        self.assertIn("macro_f1", result)


if __name__ == "__main__":
    unittest.main()
