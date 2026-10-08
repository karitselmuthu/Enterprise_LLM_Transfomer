"""Load each generation behind one local prediction interface."""

import json
from pathlib import Path


def load_predictor(path: Path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    version = metadata.get("version")
    if version == "v1":
        from src.baselines.traditional import predict
        return metadata, lambda text: predict(metadata, text)
    if version == "v2":
        from src.baselines.word2vec import predict
        return metadata, lambda text: predict(metadata, text)
    if version in ("v3", "v4", "v5"):
        import torch
        from src.preprocessing.vocabulary import encode
        from src.training.train_neural import make_model

        model = make_model(metadata)
        state = torch.load(path.parent / metadata["weights_file"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        def predict_neural(text: str) -> dict:
            token_ids = torch.tensor([encode(text, metadata["vocabulary"], metadata["max_length"])])
            with torch.inference_mode():
                probabilities = torch.softmax(model(token_ids)[0], dim=0).tolist()
            scores = dict(zip(metadata["labels"], probabilities))
            return {"label": max(scores, key=scores.get), "scores": scores}

        return metadata, predict_neural
    if version in ("v6", "v7"):
        from src.inference.pretrained import load_pretrained_predictor
        return metadata, load_pretrained_predictor(metadata, path)
    raise ValueError(f"Unsupported model version: {version}")
