# Lecture 2 tests

`test_components.py` checks causal masking, cached decoding, attention variants, RoPE, MoE routing, sampling, and review transactions. `test_workflow_api.py` checks classification, pending drafts, separate reviewer authorization, approval, and one local ticket response. Run them with the full suite:

```bash
python -m unittest discover -s tests -v
```
