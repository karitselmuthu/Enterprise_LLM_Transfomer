import unittest

from src.baselines.traditional import fit, predict
from src.evaluation.evaluate import evaluate
from src.preprocessing.tokenizer import tokenize
from src.training.train import split_rows


class V1Tests(unittest.TestCase):
    def test_tokenizer_normalizes_case_and_punctuation(self):
        self.assertEqual(tokenize("VPN, vpn! Café_42"), ["vpn", "vpn", "café", "42"])

    def test_split_is_reproducible_and_disjoint(self):
        rows = [{"id": f"A{i}", "text": "alpha", "label": "a"} for i in range(5)]
        rows += [{"id": f"B{i}", "text": "beta", "label": "b"} for i in range(5)]
        train, test = split_rows(rows)
        self.assertEqual({row["id"] for row in test}, {row["id"] for row in split_rows(rows)[1]})
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in test})
        self.assertEqual({row["label"] for row in test}, {"a", "b"})

    def test_classifier_and_held_out_metrics(self):
        rows = [
            {"id": "1", "text": "vpn network vpn", "label": "network"},
            {"id": "2", "text": "password login password", "label": "access"},
        ]
        model = fit(rows)
        model["test_ids"] = ["3", "4"]
        self.assertEqual(predict(model, "vpn network")["label"], "network")
        test = [
            {"id": "3", "text": "vpn", "label": "network"},
            {"id": "4", "text": "password", "label": "access"},
        ]
        result = evaluate(model, test)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["per_class"]["network"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
