"""Small skip-gram Word2Vec with negative sampling for teaching purposes.

The input embedding predicts nearby context words. Only training tickets may be
passed here; held-out ticket text must never influence the vectors.
"""

from collections import Counter
import math
import random

from src.preprocessing.tokenizer import tokenize


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def train_embeddings(
    texts: list[str],
    dimensions: int = 24,
    window: int = 2,
    epochs: int = 40,
    negatives: int = 4,
    learning_rate: float = 0.025,
    seed: int = 42,
) -> dict:
    """Train skip-gram vectors with unigram^0.75 negative sampling."""
    if dimensions < 1 or window < 1 or epochs < 1 or negatives < 1 or learning_rate <= 0:
        raise ValueError("Word2Vec hyperparameters must be positive")
    sentences = [tokenize(text) for text in texts]
    frequencies = Counter(word for sentence in sentences for word in sentence)
    vocabulary = sorted(frequencies)
    if len(vocabulary) < 2:
        raise ValueError("Word2Vec needs at least two distinct tokens")
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    indexed = [[word_to_index[word] for word in sentence] for sentence in sentences]
    pairs = [
        (sentence[position], sentence[context])
        for sentence in indexed
        for position in range(len(sentence))
        for context in range(max(0, position - window), min(len(sentence), position + window + 1))
        if context != position
    ]
    if not pairs:
        raise ValueError("Word2Vec needs at least one context pair")

    rng = random.Random(seed)
    scale = 0.5 / dimensions
    input_vectors = [[rng.uniform(-scale, scale) for _ in range(dimensions)] for _ in vocabulary]
    output_vectors = [[0.0] * dimensions for _ in vocabulary]
    weights = [frequencies[word] ** 0.75 for word in vocabulary]
    negatives_pool = list(range(len(vocabulary)))

    for epoch in range(epochs):
        rng.shuffle(pairs)
        rate = learning_rate * (1 - 0.9 * epoch / epochs)
        for center, context in pairs:
            samples = [(context, 1)]
            for _ in range(negatives):
                sampled = rng.choices(negatives_pool, weights=weights, k=1)[0]
                if sampled != context:
                    samples.append((sampled, 0))
            for target, actual in samples:
                source_vector = input_vectors[center]
                target_vector = output_vectors[target]
                dot = sum(a * b for a, b in zip(source_vector, target_vector))
                gradient = rate * (actual - _sigmoid(dot))
                old_source = source_vector.copy()
                for dimension in range(dimensions):
                    source_vector[dimension] += gradient * target_vector[dimension]
                    target_vector[dimension] += gradient * old_source[dimension]

    return {
        "algorithm": "skip_gram_negative_sampling",
        "dimensions": dimensions,
        "window": window,
        "epochs": epochs,
        "negatives": negatives,
        "learning_rate": learning_rate,
        "seed": seed,
        "vectors": {word: input_vectors[index] for index, word in enumerate(vocabulary)},
    }


def document_vector(text: str, vectors: dict[str, list[float]], dimensions: int) -> list[float]:
    """Average known word vectors; unseen-only text maps to a zero vector."""
    words = [word for word in tokenize(text) if word in vectors]
    if not words:
        return [0.0] * dimensions
    return [sum(vectors[word][index] for word in words) / len(words) for index in range(dimensions)]


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def nearest_words(embedding: dict, word: str, count: int = 5) -> list[tuple[str, float]]:
    """Inspect learned neighborhoods, which may be noisy on tiny corpora."""
    vectors = embedding["vectors"]
    if word not in vectors:
        raise ValueError(f"Unknown word: {word}")
    neighbors = [(candidate, cosine(vectors[word], vector)) for candidate, vector in vectors.items() if candidate != word]
    return sorted(neighbors, key=lambda item: (-item[1], item[0]))[:count]
