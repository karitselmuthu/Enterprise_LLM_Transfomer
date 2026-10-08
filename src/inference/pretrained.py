"""Local-only V6/V7 inference; prediction never downloads model files."""

from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def load_pretrained_predictor(metadata: dict, metadata_path: Path):
    version = metadata["version"]
    directory = metadata_path.parent / metadata["base_dir" if version == "v6" else "model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if version == "v6":
        model = AutoModel.from_pretrained(directory, local_files_only=True)
        centroids = {label: torch.tensor(vector) for label, vector in metadata["centroids"].items()}
    else:
        model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
    model.eval()

    def predict(text: str) -> dict:
        batch = tokenizer(text, truncation=True, max_length=metadata["max_length"], return_tensors="pt")
        with torch.inference_mode():
            if version == "v6":
                feature = F.normalize(model(**batch).last_hidden_state[:, 0, :][0], dim=0)
                scores = {label: F.cosine_similarity(feature, centroid, dim=0).item()
                          for label, centroid in centroids.items()}
            else:
                probabilities = torch.softmax(model(**batch).logits[0], dim=0).tolist()
                scores = dict(zip(metadata["labels"], probabilities))
        return {"label": max(scores, key=scores.get), "scores": scores}

    return predict
