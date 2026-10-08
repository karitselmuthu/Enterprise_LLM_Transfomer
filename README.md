# Enterprise Ticket Transformer

This project joins two lectures in one local support-ticket workflow. **Lecture 1** implements V1–V7 classifiers and a V8 API. **Lecture 2** implements a small causal language model with MoE, MHA/MQA/GQA, RoPE, KV caching, and sampling exercises, then connects a trained demo decoder to draft review. The supplied tickets and replies are synthetic learning examples, not evidence of enterprise performance. Read [why the classifier generations exist](docs/generations.md) alongside the code. The project is licensed under [MIT](LICENSE).

| Version | Implementation | What it adds |
| --- | --- | --- |
| V1 | Token counts + Naive Bayes | A transparent word-count baseline |
| V2 | Skip-gram Word2Vec + class centroids | Learned word vectors from nearby words |
| V3 | LSTM | Word order and recurrent state |
| V4 | LSTM + additive attention | A learned weight for each token |
| V5 | Scratch Transformer encoder | Q/K/V self-attention and sinusoidal positions |
| V6 | Frozen pretrained BERT + class centroids | Language representations learned elsewhere |
| V7 | Fine-tuned pretrained BERT | Update pretrained weights for ticket labels |
| V8 | FastAPI + Docker | Model loading, input validation, API key, health check |

The scratch neural stages use [PyTorch's LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) for V3/V4 and explicit Q/K/V projections in `src/transformer/attention.py` for V5. V6/V7 use the small [bert-tiny model](https://huggingface.co/prajjwal1/bert-tiny) by default. It is an educational choice; select and approve a model separately for enterprise use.

## Lecture paths and repository layout

- [Lecture 1: Transformers and ticket classification](docs/lecture_01_transformers/README.md) covers tokenization → embeddings → Word2Vec → LSTM → attention → Transformer encoder, then model evaluation and serving.
- [Lecture 2: Large Language Models](docs/lecture_02_llm/README.md) implements architectures → mixture of experts → MHA/MQA/GQA → RoPE → context and KV caching → sampling → response drafting.

```text
src/
  preprocessing/ embeddings/ baselines/ transformer/  # Lecture 1 models
  llm/                                                # Lecture 2 decoder and workflow
    architectures/ moe/ attention/ positional_encoding/
    context/ sampling/ inference/
  training/ evaluation/ inference/                    # Existing shared workflows
configs/
  transformer/ llm/                                   # Lecture-specific settings
  *.json                                              # Existing locked splits and examples
docs/
  lecture_01_transformers/ lecture_02_llm/
tests/
  transformer/ llm/                                   # New phase tests live here
  test_*.py                                           # Existing Lecture 1 tests
```

Existing classifier module paths and split manifests remain usable. The new end-to-end demo uses a scratch V5 Transformer classifier and a tiny decoder trained on fictional response examples. Its SQLite response table is a **local ticket-system adapter**; no external support platform is connected. A later training lecture can improve the decoder after these mechanisms are understood.

## Run the two-lecture workflow

First install the development requirements from the next section, then train both synthetic demo artifacts:

```bash
.venv/bin/python -m scripts.bootstrap_workflow
TICKET_MODE=demo \
TICKET_MODEL_PATH=models/transformer_demo.json \
TICKET_LLM_PATH=models/llm_demo.pt \
TICKET_REVIEW_DB=data/runtime/tickets.sqlite3 \
TICKET_API_KEY=demo-user-key \
TICKET_REVIEWER_API_KEY=demo-review-key \
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal, create a pending draft, inspect it, then approve it. The returned draft ID replaces `DRAFT_ID` below.

```bash
curl -H 'X-API-Key: demo-user-key' -H 'Content-Type: application/json' \
  -d '{"ticket_id":"INC-DEMO-1","text":"VPN connection drops during calls"}' \
  http://127.0.0.1:8000/drafts
curl -H 'X-Reviewer-Key: demo-review-key' \
  http://127.0.0.1:8000/drafts/DRAFT_ID
curl -H 'X-Reviewer-Key: demo-review-key' -H 'Content-Type: application/json' \
  -d '{"reviewer":"support-agent","edited_response":"We will review the VPN connection and follow up."}' \
  http://127.0.0.1:8000/drafts/DRAFT_ID/approve
curl -H 'X-Reviewer-Key: demo-review-key' \
  http://127.0.0.1:8000/tickets/INC-DEMO-1/responses
```

Before approval, the local ticket has no response record. Rejection leaves it empty; repeated approval returns 409. The separate reviewer key gates the decision endpoint. This is a learning workflow: the demo model memorizes simple synthetic reply patterns, the reviewer name is not tied to an identity provider, and production mode refuses to start the LLM workflow. The [Lecture 2 guide](docs/lecture_02_llm/README.md) shows the modules, tests, and limits.

## Run locally

Use Python 3.12. The commands below use `uv` because this workstation has no `python3.12` shell command.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

The saved sample split in `configs/sample_split.json` has 32 training IDs and 8 test IDs. Its dataset hash prevents accidentally evaluating modified tickets against old artifacts. Build a separate split once for any new dataset; keep its test IDs fixed while developing models. CI runs the unit tests and an audit of the application sample; pretrained weights are not downloaded by CI.

```bash
.venv/bin/python -m src.training.train                         # V1
.venv/bin/python -m src.training.train_v2                      # V2
.venv/bin/python -m src.training.train_neural --version v3     # LSTM
.venv/bin/python -m src.training.train_neural --version v4     # LSTM + attention
.venv/bin/python -m src.training.train_neural --version v5     # Transformer
.venv/bin/python -m src.training.prepare_pretrained           # Download once
.venv/bin/python -m src.training.train_pretrained --version v6 # Frozen encoder
.venv/bin/python -m src.training.train_pretrained --version v7 # Fine-tuning
.venv/bin/python -m src.evaluation.compare_all
.venv/bin/python -m src.inference.predict --model models/v5.json "VPN disconnects"
.venv/bin/python -m src.embeddings.inspect vpn --model models/v2.json
```

`prepare_pretrained` needs network access. It stores a local copy in `models/pretrained_base`; V6/V7 training and prediction use that copy without downloading at inference time. Its `--model-id` and `--revision` flags let you select and pin another approved checkpoint. Model files are generated locally and excluded from Git.

## Sample comparison

All stages use the same eight held-out synthetic tickets. These scores only confirm that each workflow runs; two tickets per class cannot rank production models reliably.

| Stage | Accuracy | Macro F1 |
| --- | ---: | ---: |
| V1 | 0.500 | 0.458 |
| V2 | 0.250 | 0.167 |
| V3 | 0.500 | 0.435 |
| V4 | 0.500 | 0.458 |
| V5 | 0.375 | 0.333 |
| V6 | 0.500 | 0.542 |
| V7 | 0.125 | 0.063 |

See [V1/V2 mistakes](reports/sample_error_analysis.md). The full comparison command prints mistake IDs for every version. In particular, V7's low sample result shows why the API must serve a deliberately selected model rather than defaulting to the latest generation.

An [expanded 400-ticket synthetic set](data/samples/files/README.md) is also available with incident links, an 80-ticket grouped split, and a separate challenge set. It is useful for practice and pipeline checks; its templated wording causes substantial near-duplicate overlap between training and test tickets. It is not a substitute for representative enterprise data.

The [application ticket examples](data/samples/applications/README.md) contain 80 newly written fictional tickets across Okta, Jira, Microsoft 365, and Workday, with a combined CSV that the training commands can read directly. Their [V1–V7 comparison](reports/application_comparison.md) includes per-class and per-application results and every held-out mistake.

## Use your ticket data

Prepare a de-identified UTF-8 CSV with unique `id`, nonempty `text`, and consistent `label` values. Put it under `data/processed/`, which is ignored by Git. The split helper supports incident grouping with `--groups` and a separate validation partition with `--validation-fraction`. Audit the data and linked incidents before fitting a model. The full procedure, including a single final test evaluation and promotion review, is in [real-ticket evaluation and model promotion](docs/real_data_and_promotion.md).

```bash
.venv/bin/python -m src.training.split --data data/processed/tickets.csv --groups data/processed/incident_groups.csv --validation-fraction 0.2 --output configs/private/real_split.json
.venv/bin/python -m src.evaluation.data_audit --data data/processed/tickets.csv --groups data/processed/incident_groups.csv --split configs/private/real_split.json --output reports/private/data_audit.json
.venv/bin/python -m src.training.train --data data/processed/tickets.csv --split configs/private/real_split.json --model models/tickets_v1.json
.venv/bin/python -m src.training.train_neural --version v5 --data data/processed/tickets.csv --split configs/private/real_split.json --model models/tickets_v5.json
```

Pass the same `--data` and `--split` paths to every stage. V2, V6, and V7 accept those flags as well. Evaluate the validation partition with `src.evaluation.evaluate --data ... --model ... --partition validation` or compare all model paths with `src.evaluation.compare_all --data ... --models ... --partition validation`. Use the test partition only after choosing a candidate.

## Serve V8

The API loads the selected artifact once at startup. Its default `TICKET_MODE=demo` requires `TICKET_API_KEY`; setting `TICKET_ALLOW_UNAUTHENTICATED=1` is intended only for local exploration. `TICKET_MODE=production` additionally requires a private [promotion manifest](docs/real_data_and_promotion.md) matching a real-data model and both evaluation reports. `POST /predict` accepts `{ "text": "..." }`; `GET /health` reports readiness and the loaded generation. The response calls cosine scores *similarities* for V2/V6 and model probabilities for other stages; neither should be treated as calibrated confidence without validation.

```bash
TICKET_API_KEY=change-me TICKET_MODEL_PATH=models/v1.json .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
curl -H 'X-API-Key: change-me' -H 'Content-Type: application/json' \
  -d '{"text":"VPN disconnects during meetings"}' http://127.0.0.1:8000/predict
```

The Dockerfile trains the synthetic V5 classifier and toy decoder while building the image. Set both API keys at runtime and mount `/app/data/runtime` if you want to keep local review records between containers. Mount separately trained artifacts and set their paths to try another classifier or decoder. Use real secrets through your deployment platform and TLS at the ingress.

```bash
docker build -t ticket-classifier .
docker run --rm -p 8000:8000 -e TICKET_API_KEY=demo-user-key -e TICKET_REVIEWER_API_KEY=demo-review-key ticket-classifier
```

The API and container configuration are a local demonstration. The Docker image has not been built here because a daemon is unavailable. Enterprise launch still needs real-data evaluation of both classification and replies, approved model weights and licensing, authenticated reviewer identities, a support-system connector, secret management, monitoring, and an incident/rollback plan. No live deployment is included here.
