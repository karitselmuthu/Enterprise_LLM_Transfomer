# V1–V7 application ticket comparison

## Evaluation contract

- Data: `data/samples/applications/all_applications.csv` (80 fictional tickets)
- Locked split: `configs/application_sample_split.json`, dataset SHA-256 `f49d6c351e0bc791844957beb91fbcd4c95e58f036b4252b5225b757bbfe9778`
- Training: 64 tickets, 16 per label. Test: 16 tickets, 4 per label. Every model records the same test IDs and dataset hash.
- Test application counts: Jira 5, Microsoft 365 4, Okta 5, Workday 2.
- Machine-readable scores and mistake IDs: [application_all_metrics.json](application_all_metrics.json).

## Overall results

| Version | Correct / 16 | Accuracy | Macro F1 | Mistakes |
| --- | ---: | ---: | ---: | ---: |
| V1 token counts | 15 | 0.938 | 0.937 | 1 |
| V2 Word2Vec | 10 | 0.625 | 0.611 | 6 |
| V3 LSTM | 9 | 0.563 | 0.566 | 7 |
| V4 LSTM + attention | 10 | 0.625 | 0.624 | 6 |
| V5 scratch Transformer | 12 | 0.750 | 0.753 | 4 |
| V6 frozen pretrained encoder | 13 | 0.813 | 0.804 | 3 |
| V7 fine-tuned encoder | 6 | 0.375 | 0.278 | 10 |

Per-class F1 (each class has only four test tickets):

| Label | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Access | 1.000 | 0.857 | 0.500 | 0.857 | 0.667 | 1.000 | 0.667 |
| Hardware | 1.000 | 0.571 | 0.444 | 0.571 | 0.600 | 0.667 | 0.444 |
| Network | 0.889 | 0.615 | 0.750 | 0.400 | 0.889 | 0.750 | 0.000 |
| Software | 0.857 | 0.400 | 0.571 | 0.667 | 0.857 | 0.800 | 0.000 |

Correct predictions by application (the denominator differs by application):

| Application | Test rows | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jira | 5 | 5 | 4 | 4 | 4 | 5 | 4 | 2 |
| Microsoft 365 | 4 | 4 | 2 | 1 | 2 | 3 | 3 | 1 |
| Okta | 5 | 4 | 2 | 2 | 3 | 2 | 5 | 3 |
| Workday | 2 | 2 | 2 | 2 | 1 | 2 | 1 | 0 |

## Ticket-level mistakes

An omitted version classified that ticket correctly. Labels after arrows are the wrong predictions.

| Ticket | Application | Expected | Wrong predictions |
| --- | --- | --- | --- |
| JI005 | Jira | access | V2→hardware, V3→hardware |
| JI009 | Jira | hardware | V6→network |
| JI011 | Jira | network | V7→hardware |
| JI013 | Jira | network | V4→hardware, V7→hardware |
| JI016 | Jira | software | V7→hardware |
| M3001 | Microsoft 365 | access | V3→software, V5→hardware, V7→hardware |
| M3009 | Microsoft 365 | hardware | V2→network, V6→software |
| M3015 | Microsoft 365 | network | V3→access, V4→hardware, V7→hardware |
| M3019 | Microsoft 365 | software | V2→network, V3→hardware, V4→hardware, V7→hardware |
| OK005 | Okta | access | V4→hardware, V5→hardware |
| OK007 | Okta | hardware | V2→network, V3→network, V5→network |
| OK010 | Okta | hardware | V3→access |
| OK016 | Okta | software | V2→network, V7→hardware |
| OK020 | Okta | software | V1→network, V2→network, V3→hardware, V4→hardware, V5→hardware, V7→hardware |
| WD005 | Workday | access | V7→hardware |
| WD015 | Workday | network | V4→hardware, V6→software, V7→hardware |

## Review notes

- **OK020** is missed by six of seven models. Its text says that an Okta sign-in button stopped working *after a browser update*; the sample label is `software`. Several models routed it to `network` or `hardware`. This is a recurring boundary between the visible sign-in symptom and the stated software change.
- **M3019** is missed by four models. Its OneDrive sync-conflict text is labeled `software`, but predictions include `network` and `hardware`. **OK007**, a phone with a shattered screen used for Okta Verify, is labeled `hardware` and was routed to `network` by three models.
- **V7** has zero network and software F1 on this split and routes many mistakes to `hardware`. The four-class training set has 16 examples per class; the result shows this fine-tuning run is unreliable on these examples. It does not establish that fine-tuning is generally worse.
- Workday has only **two** test tickets, so its per-application counts are particularly unstable. One changed prediction moves its apparent accuracy by 50 percentage points.

These are fictional examples with a tiny holdout. They verify that the seven pipelines can be compared on identical IDs, but **do not support a production model choice**. Do not tune on these 16 test tickets. The next evaluation dataset needs real de-identified tickets, approved labels, incident grouping, near-duplicate review, a development split, and a final untouched test set.

## Reproduce

```bash
.venv/bin/python -m src.evaluation.compare_all \
  --data data/samples/applications/all_applications.csv \
  --models models/application_v1.json models/application_v2.json models/application_v3.json models/application_v4.json models/application_v5.json models/application_v6.json models/application_v7.json \
  --output reports/application_all_metrics.json
```
