import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from api.main import create_app
from src.baselines.traditional import fit
from src.baselines.lstm import LSTMClassifier
from src.preprocessing.vocabulary import build_vocabulary, encode
from src.transformer.attention import SelfAttention
from src.transformer.model import TransformerClassifier
from src.transformer.positional_encoding import PositionalEncoding


class NeuralTests(unittest.TestCase):
    def test_vocabulary_uses_training_words_and_preserves_order(self):
        vocabulary = build_vocabulary(["vpn login"])
        self.assertNotIn("unseen", vocabulary)
        self.assertEqual(encode("login vpn", vocabulary, 4)[:2],
                         [vocabulary["login"], vocabulary["vpn"]])

    def test_lstm_attention_and_transformer_shapes(self):
        tokens = torch.tensor([[2, 3, 0, 0], [3, 2, 2, 0]])
        for attention in (False, True):
            output = LSTMClassifier(4, 2, embedding_dim=8, hidden_dim=8,
                                    attention=attention)(tokens)
            self.assertEqual(tuple(output.shape), (2, 2))
        transformer = TransformerClassifier(4, 2, dimension=8, heads=2,
                                             layers=1, max_length=4)
        self.assertEqual(tuple(transformer(tokens).shape), (2, 2))
        self.assertEqual(tuple(PositionalEncoding(8, 4).encoding.shape), (1, 4, 8))
        self.assertEqual(tuple(SelfAttention(8, 2)(torch.randn(2, 4, 8), tokens.ne(0)).shape),
                         (2, 4, 8))


class APITests(unittest.TestCase):
    def test_api_requires_key_and_serves_prediction(self):
        rows = [{"text": "VPN down", "label": "network"},
                {"text": "Reset password", "label": "access"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "v1.json"
            model_path.write_text(json.dumps(fit(rows)), encoding="utf-8")
            with patch.dict(os.environ, {"TICKET_API_KEY": "test-secret", "TICKET_MODE": "demo"}):
                app = create_app(model_path)
                with TestClient(app) as client:
                    self.assertEqual(client.get("/health").json()["model_version"], "v1")
                    self.assertEqual(client.post("/predict", json={"text": "VPN down"}).status_code, 401)
                    response = client.post("/predict", headers={"X-API-Key": "test-secret"},
                                           json={"text": "VPN down"})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(response.json()["label"], ["access", "network"])
                    self.assertEqual(client.post("/predict", headers={"X-API-Key": "test-secret"},
                                                 json={"text": "   "}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
