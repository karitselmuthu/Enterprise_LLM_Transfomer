# Lecture 2 source map

The package contains a small causal decoder and a local ticket-response demo. It complements the Lecture 1 classifier in `src/transformer/`.

| Package | Phase | Code |
| --- | --- | --- |
| `architectures/` | 2.1 | Decoder blocks and next-token loss |
| `moe/` | 2.2 | Top-1 routed experts |
| `attention/` | 2.3 | Causal MHA, MQA, and GQA |
| `positional_encoding/` | 2.4 | RoPE |
| `context/` | 2.5 | KV cache inspection and incremental decoding |
| `sampling/` | 2.6 | Temperature, top-k, top-p, greedy |
| `inference/` | 2.7 | Toy training, generation, SQLite review store |

See the [Lecture 2 guide](../../docs/lecture_02_llm/README.md). The training data and responses are fictional; a generated draft always requires human review. The integration writes approved replies only to a local SQLite ticket-response table.
