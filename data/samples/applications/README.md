# Application ticket examples

These are **fictional but realistic-style** help-desk tickets, written for this project. They are not copied from actual support history and contain no intentionally included personal details. The four files cover Okta, Jira, Microsoft 365, and Workday. Each has 20 rows: five each for `access`, `hardware`, `network`, and `software`.

`application` names the affected application or workflow. `label` names the likely root cause, so a Jira ticket can correctly be labeled `network` or `hardware`. All CSVs include `id,text,label,application`; the existing trainer reads the first three columns and ignores `application`.

`all_applications.csv` combines the four files into 80 rows for a single training run. Rebuild it after editing a source file:

```bash
python3 scripts/combine_application_samples.py
```

To try the existing pipeline from the repository root:

```bash
python3 -m src.training.split --data data/samples/applications/all_applications.csv --output configs/application_sample_split.json
python3 -m src.training.train --data data/samples/applications/all_applications.csv --split configs/application_sample_split.json --model models/application_v1.json
python3 -m src.evaluation.evaluate --data data/samples/applications/all_applications.csv --model models/application_v1.json
```

V2–V7 can use the **same saved split**. Use the project Python environment for the neural stages; V6/V7 also need the local `models/pretrained_base` checkpoint prepared as described in the root README.

```bash
DATA=data/samples/applications/all_applications.csv
SPLIT=configs/application_sample_split.json
python3 -m src.training.train_v2 --data "$DATA" --split "$SPLIT" --model models/application_v2.json
for version in v3 v4 v5; do
  .venv/bin/python -m src.training.train_neural --version "$version" --data "$DATA" --split "$SPLIT" --model "models/application_${version}.json"
done
for version in v6 v7; do
  .venv/bin/python -m src.training.train_pretrained --version "$version" --data "$DATA" --split "$SPLIT" --base-dir models/pretrained_base --model "models/application_${version}.json"
done
```

All seven local artifacts were trained against this split and load successfully. They are generated files under `models/` and are ignored by Git. The [V1–V7 comparison and mistake review](../../../reports/application_comparison.md) records the held-out results.

These examples are deliberately balanced and small. They demonstrate schema and workflow behavior; they are not a representative enterprise benchmark. Review the labels with the service-desk owner before using similar categories for real tickets.
