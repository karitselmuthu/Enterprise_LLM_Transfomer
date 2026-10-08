# Synthetic sample error analysis

This report uses the locked `configs/sample_split.json` split: 32 training tickets, 8 held-out tickets, two per class. It is a teaching check, not an enterprise benchmark.

| Model | Correct | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| V1 token counts + Naive Bayes | 4/8 | 0.50 | 0.458 |
| V2 Word2Vec + class centroids | 2/8 | 0.25 | 0.167 |

## V1 mistakes

| ID | Actual → predicted | Training-vocabulary words in the held-out text | Observation |
| --- | --- | --- | --- |
| A04 | access → software | `please`, `to`, `the` | The informative words, including `access`, were absent from V1 training tickets. |
| A08 | access → software | `please`, `a` | Almost no category evidence survived tokenization and vocabulary filtering. |
| H04 | hardware → access | `my`, `monitor`, `not`, `on` | `monitor` was known, but their combined class likelihood favored `access`. |
| N06 | network → hardware | `the`, `in`, `our`, `is` | None of the network-specific words were seen in training. |

V2 makes the same four mistakes, plus H06 (`hardware` → `access`) and N08 (`network` → `access`). H06 has only `is` in its training vocabulary; N08 has `network` and `is`. Averaging vectors cannot recover unseen words, and Word2Vec trained on 32 short tickets has very little context from which to learn useful relationships.

## What to do with real tickets

1. Agree on label definitions and de-identify ticket text.
2. Put representative examples from each label in a new CSV with `id`, `text`, and `label` columns. Keep repeated or near-duplicate incident threads together before creating a split.
3. Create a new split manifest **once** and keep its test IDs untouched while developing V1 and V2.
4. Train both models on the same training IDs and compare macro F1, per-class metrics, and mistake IDs. Review mistakes with someone who understands the ticket labels.
5. If a separate unlabeled enterprise corpus is available, Word2Vec may be trained on its text, provided held-out evaluation tickets and duplicate incident threads are excluded. Record that corpus and its provenance before comparing models.

The sample metrics cannot answer whether Word2Vec improves enterprise classification. They do show that the evaluation code exposes errors and that adding an embedding layer alone does not guarantee an improvement.
