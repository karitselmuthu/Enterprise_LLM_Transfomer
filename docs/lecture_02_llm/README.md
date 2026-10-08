# Lecture 2: Large Language Models

**Status:** all seven educational phases have runnable local implementations. The integrated ticket workflow is a synthetic demonstration. It does not send messages to an external support platform and does not support production promotion.

## Application flow

```text
Fictional support ticket
  → V5 Transformer classifier (Lecture 1)
  → Label and original ticket text
  → Tiny causal decoder (Lecture 2)
  → Pending suggested response in SQLite
  → Human review, optional edit, and approval or rejection
  → Approved response in the local ticket-system table
```

The response table is the local adapter for the final stage. The demo cannot write a response before approval. Rejection and duplicate approval do not create a ticket response. No ServiceNow, Jira Service Management, or other external connector is configured.

## Phases and source

| Phase | Concept | Implementation | Verification |
| --- | --- | --- | --- |
| 2.1 | Encoder versus decoder | Lecture 1's bidirectional classifier in `src/transformer/`; causal next-token model in `src/llm/architectures/decoder.py` | Future token changes do not alter earlier logits |
| 2.2 | Mixture of Experts | `src/llm/moe/routing.py` top-1 expert router with capacity and overflow | Routing counts and overflow |
| 2.3 | MHA / MQA / GQA | `src/llm/attention/variants.py` with configurable KV head count | Shapes, causal masking, parameter counts |
| 2.4 | RoPE | `src/llm/positional_encoding/rope.py` rotates queries and keys | Position dependence |
| 2.5 | Context length | `src/llm/context/cache.py` and incremental attention caches | Cached logits match uncached logits |
| 2.6 | Sampling | `src/llm/sampling/strategies.py` temperature, top-k, top-p, greedy | Filtering and invalid parameter checks |
| 2.7 | Integration | `src/llm/inference/` and `api/main.py` | Draft, reject, approve, and local ticket update tests |

The demo decoder uses grouped-query attention (`heads=4`, `kv_heads=2`) and a dense feed-forward block. Change `configs/llm/demo.json` to `kv_heads=4` for MHA, `kv_heads=1` for MQA, or `experts=2` to train with the lightweight MoE. The modules are teaching implementations and are not optimized for long contexts or production throughput.

## Run it

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.bootstrap_workflow
```

The bootstrap trains V5 on `data/samples/applications/all_applications.csv` using the locked application split and trains the decoder on fictional ticket texts and generic response patterns in `data/samples/tickets.csv`. Artifacts are written under ignored `models/`. Use the [main README](../../README.md#run-the-two-lecture-workflow) to start the API and exercise approval. `data/runtime/` is ignored because its SQLite file contains ticket text and reviewer decisions.

## Evaluation and limits

The V5 application sample test has 16 synthetic tickets; its result is recorded in [the Lecture 1 comparison](../../reports/application_comparison.md). The decoder training loss only measures how well it fits four generic reply patterns. It does not measure whether drafts are useful, correct, safe, or appropriate for real customers.

The API requires a separate reviewer key, but the reviewer name is a caller-provided string. Production mode deliberately refuses to enable these LLM routes. Before connecting a real ticket system, obtain representative de-identified ticket/response examples, define response-quality criteria and prohibited content, choose and approve a suitable generative checkpoint, authenticate reviewer identity, define retention and audit rules, test the external adapter, and evaluate the whole flow on held-out incidents.
