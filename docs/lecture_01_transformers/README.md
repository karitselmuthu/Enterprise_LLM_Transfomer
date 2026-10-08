# Lecture 1: Transformers and ticket classification

The working V1–V8 sequence covers tokenization, embeddings, Word2Vec, LSTM, attention, a scratch Transformer encoder, pretrained encoders, and a classification API.

- [Main README and run commands](../../README.md)
- [Why each generation exists](../generations.md)
- [Real-ticket evaluation and model promotion](../real_data_and_promotion.md)
- [Application sample and V1–V7 comparison](../../data/samples/applications/README.md)

Lecture 1 code remains in `src/preprocessing/`, `src/embeddings/`, `src/baselines/`, `src/transformer/`, `src/training/`, `src/evaluation/`, `src/inference/`, and `api/`. Shared training and evaluation modules can be extended by later lectures without moving these imports.
