"""Multinomial Naive Bayes over token counts, implemented with the stdlib."""

from collections import Counter
import math

from src.preprocessing.tokenizer import tokenize


def fit(rows: list[dict[str, str]], alpha: float = 1.0) -> dict:
    if not rows:
        raise ValueError("Training rows cannot be empty")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    labels = sorted({row["label"] for row in rows})
    document_counts = Counter(row["label"] for row in rows)
    word_counts = {label: Counter() for label in labels}
    for row in rows:
        word_counts[row["label"]].update(tokenize(row["text"]))
    vocabulary = sorted({word for counts in word_counts.values() for word in counts})
    if not vocabulary:
        raise ValueError("Training rows contain no word tokens")

    return {
        "version": "v1",
        "alpha": alpha,
        "labels": labels,
        "document_counts": dict(document_counts),
        "word_counts": {label: dict(word_counts[label]) for label in labels},
        "vocabulary": vocabulary,
    }


def predict(model: dict, text: str) -> dict:
    """Return the winning label and normalized scores for all labels.

    Scores are model probabilities, not calibrated confidence estimates.
    Unknown words are ignored; an all-unknown input follows the class priors.
    """
    labels = model["labels"]
    vocab = set(model["vocabulary"])
    words = Counter(word for word in tokenize(text) if word in vocab)
    total_documents = sum(model["document_counts"].values())
    log_scores = {}
    for label in labels:
        counts = model["word_counts"][label]
        denominator = sum(counts.values()) + model["alpha"] * len(vocab)
        score = math.log(model["document_counts"][label] / total_documents)
        score += sum(
            frequency * math.log((counts.get(word, 0) + model["alpha"]) / denominator)
            for word, frequency in words.items()
        )
        log_scores[label] = score

    peak = max(log_scores.values())
    weights = {label: math.exp(score - peak) for label, score in log_scores.items()}
    total_weight = sum(weights.values())
    scores = {label: weights[label] / total_weight for label in labels}
    return {"label": max(labels, key=lambda label: (scores[label], label)), "scores": scores}
