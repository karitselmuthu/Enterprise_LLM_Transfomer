# Expanded synthetic ticket sample

This is **synthetic practice data**, not representative enterprise tickets. The original 40 tickets in `../tickets.csv` are preserved exactly in `tickets_sample.csv`; 360 templated tickets were added.

| File | Contents | Use |
| --- | --- | --- |
| `tickets_sample.csv` | 400 rows, 100 for each of access, hardware, network, software | Training and a fixed incident-grouped test split |
| `challenge_set.csv` | 20 separate boundary cases with rationales | Final challenge review; never train or tune on these rows |
| `incidents.csv` | Six synthetic incidents | Context for the linked ticket groups |
| `ticket_incidents.csv` | 73 ticket-to-incident links | Group key while creating a split |

The generator lives at `../../../scripts/generate_samples.py`. Its output was checked against all four CSVs. Sample label definitions, acceptance targets, checkpoint approval, deployment platform, and secret-name templates are in `../../../configs/sample_bundle/`. Those templates have placeholder values and no production approval.

To use these tickets with the current pipeline:

```bash
python3 -m src.training.split --data data/samples/files/tickets_sample.csv --groups data/samples/files/ticket_incidents.csv --output configs/expanded_sample_split.json
python3 -m src.training.train --data data/samples/files/tickets_sample.csv --split configs/expanded_sample_split.json --model models/expanded_v1.json
python3 -m src.evaluation.evaluate --data data/samples/files/tickets_sample.csv --model models/expanded_v1.json
```

`configs/expanded_sample_split.json` is already saved: 320 training and 80 test tickets, 20 test tickets per class, with no linked incident crossing the split.

## Validation limits

The V1 baseline scores **0.95 accuracy and 0.949 macro F1** on that 80-ticket test set; V2 scores **0.863 accuracy and 0.863 macro F1**. These results are inflated by templated wording: 13/80 test tickets have a same-label training ticket with text similarity at least 0.85 by Python's `SequenceMatcher`. This is a screening heuristic, not a formal duplicate audit. The data is useful for learning and pipeline checks, but cannot establish enterprise quality or satisfy the sample near-duplicate acceptance target.

On the 19 challenge rows with the four training labels, V1 gets 12 correct (0.632 accuracy). The twentieth case has label `needs_triage`, which the current four-class models cannot emit; it needs a separately validated abstain policy before the full challenge set can be scored as intended. Do not tune on the challenge set.
