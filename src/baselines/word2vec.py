"""V2: classify tickets using mean Word2Vec vectors and class centroids."""

from collections import Counter

from src.embeddings.word2vec import cosine, document_vector, train_embeddings


def fit(rows: list[dict[str, str]], **embedding_options) -> dict:
    if not rows:
        raise ValueError("Training rows cannot be empty")
    embedding = train_embeddings([row["text"] for row in rows], **embedding_options)
    dimensions = embedding["dimensions"]
    labels = sorted({row["label"] for row in rows})
    sums = {label: [0.0] * dimensions for label in labels}
    counts = Counter(row["label"] for row in rows)
    for row in rows:
        vector = document_vector(row["text"], embedding["vectors"], dimensions)
        for index, value in enumerate(vector):
            sums[row["label"]][index] += value
    centroids = {
        label: [value / counts[label] for value in sums[label]]
        for label in labels
    }
    return {"version": "v2", "labels": labels, "embedding": embedding, "centroids": centroids}


def predict(model: dict, text: str) -> dict:
    embedding = model["embedding"]
    vector = document_vector(text, embedding["vectors"], embedding["dimensions"])
    similarities = {label: cosine(vector, model["centroids"][label]) for label in model["labels"]}
    # Cosine values are relative similarities, not probabilities.
    return {
        "label": max(model["labels"], key=lambda label: (similarities[label], label)),
        "scores": similarities,
    }
