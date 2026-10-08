# Enterprise LLM Transformer

### From Transformer fundamentals to a reviewed support-ticket response

This educational project connects two lectures in one local application. **Lecture 1** classifies a support ticket. **Lecture 2** uses a small decoder to suggest a reply. A reviewer can edit, approve, or reject the suggestion. An approved response is written to a **local SQLite ticket-system table**.

> **Project status:** V1–V7 classifiers, the V8 FastAPI service, Lecture 2 concept implementations, and the local draft-review workflow are implemented. The included tickets and replies are fictional. No external support system is connected, and generated replies have not been evaluated for real customer use.

## 1. Project overview

Support teams receive free-text tickets about access, hardware, network, and software issues. This repository shows why each text representation and model architecture exists, then demonstrates how a generative model can assist an agent while keeping a human decision in the workflow.

```text
Lecture 1: NLP → tokenization → embeddings → Word2Vec → LSTM
           → attention → Transformer encoder → pretrained encoders → API

Lecture 2: encoder versus decoder → MoE → MHA/MQA/GQA → RoPE
           → context and KV caching → sampling → response drafting
```

The small datasets and models establish that the code paths work. They do not establish enterprise accuracy, response quality, or production readiness.

## 2. Application flow

```text
Customer support ticket
        ↓
V5 Transformer encoder (Lecture 1 demo)
        ↓
Ticket classification: access / hardware / network / software
        ↓
Tiny causal decoder (Lecture 2 demo)
        ↓
Suggested response saved as a pending draft
        ↓
Human review: edit and approve, or reject
        ↓
Approved response stored in the local ticket-system table
```

`POST /drafts` creates a suggestion but does not create a ticket response. Only approval writes to the local response table. Rejection leaves that table unchanged; a second decision on the same draft returns HTTP 409. The reviewer key is separate from the ticket-submission key. The `reviewer` field is supplied by the caller and is **not** backed by an identity provider.

The integrated demo trains a scratch V5 classifier on fictional application tickets and a tiny decoder on generic fictional reply patterns. Choosing V5 for this demo is instructional, not a production model-selection result.

## 3. Lecture 1 — Transformers

| Generation | Implementation | Concept it introduces |
| --- | --- | --- |
| V1 | Token counts and Naive Bayes | A transparent baseline |
| V2 | Skip-gram Word2Vec and class centroids | Learned word embeddings |
| V3 | LSTM | Sequence order and recurrent state |
| V4 | LSTM with additive attention | Token weighting |
| V5 | Scratch Transformer encoder | Q/K/V self-attention and positional encoding |
| V6 | Frozen pretrained BERT and class centroids | Reused pretrained representations |
| V7 | Fine-tuned pretrained BERT | Updating pretrained weights for ticket labels |
| V8 | FastAPI and Docker | Serving a selected classifier |

See [why each generation exists](docs/generations.md) and the [Lecture 1 guide](docs/lecture_01_transformers/README.md). V6 and V7 require a downloaded, approved pretrained checkpoint. Generated model artifacts are excluded from Git.

## 4. Lecture 2 — Large Language Models

| Phase | Topic | Working implementation |
| --- | --- | --- |
| 2.1 | LLM architectures | Compare the bidirectional classifier with a causal next-token decoder |
| 2.2 | Mixture of Experts | Lightweight top-1 routing with capacity and overflow |
| 2.3 | MHA, MQA, GQA | Configurable query and key/value head counts |
| 2.4 | RoPE | Rotary position embeddings on queries and keys |
| 2.5 | Context length | Incremental decoding with a KV cache |
| 2.6 | Sampling | Greedy, temperature, top-k, and top-p choices |
| 2.7 | Integration | Generate a draft and require review before a local ticket update |

The default decoder uses grouped-query attention (`heads=4`, `kv_heads=2`) and a dense feed-forward layer. In [the decoder configuration](configs/llm/demo.json), `kv_heads=4` selects MHA, `kv_heads=1` selects MQA, and `experts=2` enables the lightweight MoE during training. See the [Lecture 2 guide](docs/lecture_02_llm/README.md) for code paths, tests, and limits.

## 5. Repository structure

```text
Enterprise_LLM_Transfomer/
├── api/                         # Classification and review routes
├── configs/                     # Locked splits and model settings
│   ├── transformer/
│   └── llm/
├── data/samples/                # Fictional tickets and replies
├── docs/                        # Lecture guides and real-data procedure
│   ├── lecture_01_transformers/
│   └── lecture_02_llm/
├── reports/                     # Synthetic classification comparisons
├── scripts/                     # Two-model demo bootstrap
├── src/
│   ├── preprocessing/           # Lecture 1 tokenization
│   ├── embeddings/              # Lecture 1 embedding tools
│   ├── baselines/               # V1–V4 models
│   ├── transformer/             # Scratch Transformer encoder
│   ├── llm/                     # Lecture 2 decoder and draft workflow
│   │   ├── architectures/
│   │   ├── moe/
│   │   ├── attention/
│   │   ├── positional_encoding/
│   │   ├── context/
│   │   ├── sampling/
│   │   └── inference/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── tests/                       # Classifier, LLM, API, and review tests
│   ├── transformer/
│   └── llm/
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated models are stored under `models/`, and local review records under `data/runtime/`; both are ignored by Git. Keep private ticket data under `data/processed/`, which is also ignored.

## 6. Getting started

### Prerequisites

- Python 3.12 and `pip`, or `uv`
- Git to clone the repository
- Internet access only if you choose to download V6/V7 pretrained weights

Clone, install, and run the tests:

```bash
git clone https://github.com/karitselmuthu/Enterprise_LLM_Transfomer.git
cd Enterprise_LLM_Transfomer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

If `python3.12` is unavailable but `uv` is installed, create the environment with `uv venv --python 3.12 .venv`, then activate it and run the same install and test commands.

### Run the complete local workflow

From the repository root, with the environment activated:

```bash
python -m scripts.bootstrap_workflow
TICKET_MODE=demo \
TICKET_MODEL_PATH=models/transformer_demo.json \
TICKET_LLM_PATH=models/llm_demo.pt \
TICKET_REVIEW_DB=data/runtime/tickets.sqlite3 \
TICKET_API_KEY=demo-user-key \
TICKET_REVIEWER_API_KEY=demo-review-key \
uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

The bootstrap trains V5 using the locked split for [all_applications.csv](data/samples/applications/all_applications.csv), then trains the tiny decoder using [tickets.csv](data/samples/tickets.csv). Training outputs stay on your machine.

In a **second terminal**, create a draft. Copy the `id` from the JSON response into `DRAFT_ID` in the next commands:

```bash
curl -H 'X-API-Key: demo-user-key' -H 'Content-Type: application/json' \
  -d '{"ticket_id":"INC-DEMO-1","text":"VPN connection drops during calls"}' \
  http://127.0.0.1:8000/drafts

curl -H 'X-Reviewer-Key: demo-review-key' \
  http://127.0.0.1:8000/drafts/DRAFT_ID

curl -H 'X-Reviewer-Key: demo-review-key' \
  http://127.0.0.1:8000/tickets/INC-DEMO-1/responses
```

The response list should be empty before approval. Approve an edited response, then read the local ticket responses again:

```bash
curl -H 'X-Reviewer-Key: demo-review-key' -H 'Content-Type: application/json' \
  -d '{"reviewer":"support-agent","edited_response":"We will review the VPN connection and follow up."}' \
  http://127.0.0.1:8000/drafts/DRAFT_ID/approve

curl -H 'X-Reviewer-Key: demo-review-key' \
  http://127.0.0.1:8000/tickets/INC-DEMO-1/responses
```

To test rejection, create a **new** draft and call `POST /drafts/DRAFT_ID/reject` with `X-Reviewer-Key` and JSON such as `{"reviewer":"support-agent","reason":"Needs investigation"}`. A draft can receive only one final decision. The API also exposes `GET /health` and `POST /predict`; `/predict` takes `{"text":"VPN disconnects"}` with `X-API-Key` and returns a label, scores, model version, and score type.

### Run in Docker

The Dockerfile installs dependencies and trains the two fictional demo artifacts while building the image:

```bash
docker build -t enterprise-llm-transformer .
docker run --rm -p 8000:8000 \
  -e TICKET_API_KEY=demo-user-key \
  -e TICKET_REVIEWER_API_KEY=demo-review-key \
  enterprise-llm-transformer
```

Use the same `curl` calls against `127.0.0.1:8000`. Mount `/app/data/runtime` if review records need to survive container removal. The image has not been built in this project environment because a Docker daemon was unavailable; the Python workflow and API tests were run locally.

## 7. Train and compare classifier generations

With the environment activated, these commands train the original small synthetic sample. Each stage uses the same locked split:

```bash
python -m src.training.train
python -m src.training.train_v2
python -m src.training.train_neural --version v3
python -m src.training.train_neural --version v4
python -m src.training.train_neural --version v5
python -m src.training.prepare_pretrained
python -m src.training.train_pretrained --version v6
python -m src.training.train_pretrained --version v7
python -m src.evaluation.compare_all
```

`prepare_pretrained` downloads the default small BERT checkpoint once. V6/V7 reuse the local copy. The application-specific comparison uses 80 fictional tickets from Okta, Jira, Microsoft 365, and Workday, with 16 held-out tickets. Its [per-class results and ticket mistakes](reports/application_comparison.md) and [machine-readable metrics](reports/application_all_metrics.json) show how rankings can change on a tiny set. For example, V5 classified 12 of 16 tickets correctly in that recorded comparison. Those numbers are demonstrations, not estimates of live service-desk performance.

To train on your own data, use the [real-ticket evaluation and promotion guide](docs/real_data_and_promotion.md). It covers de-identification, consistent labels, linked incidents, fixed validation and test sets, data auditing, and classifier promotion. Keep private datasets and reports out of Git.

## 8. Evaluation and review limits

The automated suite checks the classifier pipeline, LLM component behavior, API authorization, and the draft decision transaction. Lecture 2 checks include causal masking, attention variants, RoPE, cache consistency, sampling, MoE routing, and approval/rejection behavior. Run it with `python -m unittest discover -s tests -v`.

Classifier metrics and mistake IDs are available for the included synthetic datasets. The decoder's training loss measures fit to simple reply patterns; it does **not** measure factual accuracy, relevance, safety, or customer usefulness. Human review is required even in the local demo.

The API accepts a shared reviewer key and a caller-entered reviewer name, stores ticket text in local SQLite, and has no external support-ticket connector. Production mode deliberately refuses to start the LLM draft workflow. A real deployment needs evaluated ticket and response data, approved model weights and licensing, authenticated reviewers, retention and audit rules, secret management, monitoring, and a tested support-system adapter. The classifier-only production mode additionally requires a [promotion manifest](docs/real_data_and_promotion.md); that manifest validates metadata but does not itself authenticate an approver.

## 9. Next development steps

| Step | Work needed |
| --- | --- |
| 1 | Collect approved, de-identified tickets and consistent labels; lock incident-aware evaluation splits |
| 2 | Evaluate V1–V7 against the same real-ticket test set and review mistakes |
| 3 | Collect approved response examples and define human response-quality criteria |
| 4 | Evaluate and select a suitable generative model; measure draft quality, latency, and reviewer edits |
| 5 | Integrate an authenticated reviewer identity and a real support-ticket system with audit and rollback controls |

This repository is licensed under [MIT](LICENSE).
