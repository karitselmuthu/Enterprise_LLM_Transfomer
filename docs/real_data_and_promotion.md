# Real-ticket evaluation and model promotion

The repository currently has **synthetic examples only**. Their scores demonstrate the code path, not expected performance on enterprise tickets. A production model cannot be selected until a service-desk owner supplies and approves a de-identified, consistently labeled dataset and its incident links.

## 1. Prepare private inputs

Create `data/processed/tickets.csv` with `id,text,label` columns. Each ID must be unique. Remove names, email addresses, phone numbers, account identifiers, secrets, and sensitive free text according to your organization's rules. Decide the label definitions with the people who route tickets; adjudicate ambiguous cases.

Create `data/processed/incident_groups.csv` with `ticket_id,incident_id` for tickets that share the same underlying incident. Tickets without links are treated as singletons. Linked tickets must share a label. Keep these files, private split manifests, reports, and model artifacts out of Git.

## 2. Audit and lock partitions

The split command reserves 20% for test, 20% for validation, and the remainder for training. It groups linked incidents before partitioning. The validation fraction is a fraction of all tickets. The audit checks schema, label counts, duplicate IDs, simple identifier patterns, and likely near duplicates across partitions. Review the audit manually: these checks do not establish that the text is de-identified or the labels are correct.

```bash
mkdir -p configs/private reports/private
python -m src.training.split \
  --data data/processed/tickets.csv \
  --groups data/processed/incident_groups.csv \
  --validation-fraction 0.2 \
  --output configs/private/real_split.json
python -m src.evaluation.data_audit \
  --data data/processed/tickets.csv \
  --groups data/processed/incident_groups.csv \
  --split configs/private/real_split.json \
  --output reports/private/data_audit.json
```

Inspect the audit for class imbalance, leaked identifiers, duplicated wording, and linked cases missing from the group file. Keep the locked split unchanged during development. The dataset hash in the split and model prevents silent edits to the CSV; a corrected dataset requires a new split and fresh evaluations.

## 3. Train and compare on validation

Train V1–V7 with the same `--data` and `--split` paths. The training entry points are listed in the [main README](../README.md). V6/V7 also require a pinned, reviewed pretrained checkpoint. Model fitting uses only the training partition. Compare validation macro F1, per-label recall, error IDs, latency, resource use, and the cost of false routing. Investigate label quality and class imbalance before increasing model complexity.

For each candidate:

```bash
python -m src.training.train \
  --data data/processed/tickets.csv \
  --split configs/private/real_split.json \
  --model models/real_v1.json
python -m src.evaluation.evaluate \
  --data data/processed/tickets.csv \
  --model models/real_v1.json \
  --partition validation > reports/private/real_v1_validation.json
```

When the candidate and decision criteria are frozen, run it **once** on the untouched test partition:

```bash
python -m src.evaluation.evaluate \
  --data data/processed/tickets.csv \
  --model models/real_v1.json \
  --partition test > reports/private/real_v1_test.json
```

If the test result changes the model choice, collect a new untouched test set before claiming an unbiased final estimate. Keep the report and misclassified ticket IDs private.

## 4. Approve and serve

Copy `configs/promotion.example.json` to `configs/private/promotion.json`. Set `status` to `approved` only after human review of the real-data audit, validation and test reports, operating threshold, model license, and deployment plan. Fill in the actual paths relative to the manifest, model version, dataset hash, approver, date, and target. For V6/V7, approve the pretrained checkpoint and set `checkpoint_approved` to `true`.

Production mode refuses startup without the approval manifest, matching model metadata and evaluation reports, and an API key:

```bash
TICKET_MODE=production \
TICKET_PROMOTION_MANIFEST=configs/private/promotion.json \
TICKET_MODEL_PATH=models/real_v1.json \
TICKET_API_KEY=<secret-from-your-secret-manager> \
uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

The manifest check verifies metadata consistency; it does not authenticate the approver or detect tampering. A deployment still needs TLS, access controls, logging policy, monitoring for label drift and failures, an owner for corrections, and a rollback procedure. No production target is configured in this repository.
