import tempfile
import unittest
from pathlib import Path

import torch

from src.llm.architectures.decoder import DecoderConfig, TinyDecoderLM
from src.llm.attention.variants import CausalAttention
from src.llm.context.cache import cache_length
from src.llm.inference.review import DraftStore, ReviewConflict
from src.llm.moe.routing import Top1MoE
from src.llm.positional_encoding.rope import apply_rope
from src.llm.sampling.strategies import sample_next


class Lecture2ComponentTests(unittest.TestCase):
    def test_causal_decoder_and_cached_logits(self):
        torch.manual_seed(7)
        model = TinyDecoderLM(DecoderConfig(vocab_size=24, dimension=16, heads=4,
                                            kv_heads=2, layers=2, max_context=16)).eval()
        first = torch.tensor([[1, 5, 8, 7]], dtype=torch.long)
        changed_future = torch.tensor([[1, 5, 8, 9]], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = model(first)
            changed, _ = model(changed_future)
            _, cache = model(first[:, :3])
            incremental, cache = model(first[:, 3:], cache)
        self.assertTrue(torch.allclose(logits[:, :3], changed[:, :3], atol=1e-5))
        self.assertTrue(torch.allclose(logits[:, -1:], incremental, atol=1e-5))
        self.assertEqual(cache_length(cache), 4)
        self.assertGreater(float(model.next_token_loss(first).detach()), 0)

    def test_attention_variants_rope_moe_and_sampling(self):
        x = torch.randn(2, 5, 16)
        variants = [CausalAttention(16, 4, kv) for kv in (4, 1, 2)]
        self.assertEqual([variant.variant for variant in variants], ["mha", "mqa", "gqa"])
        self.assertEqual([tuple(variant(x)[0].shape) for variant in variants], [(2, 5, 16)] * 3)
        counts = [sum(parameter.numel() for parameter in variant.parameters())
                  for variant in variants]
        self.assertGreater(counts[0], counts[2])
        self.assertGreater(counts[2], counts[1])
        positions = apply_rope(torch.ones(1, 1, 2, 4))
        self.assertFalse(torch.allclose(positions[:, :, 0], positions[:, :, 1]))
        moe = Top1MoE(16, experts=2, capacity_factor=0.5)
        output, stats = moe(x)
        self.assertEqual(tuple(output.shape), (2, 5, 16))
        self.assertEqual(sum(stats["counts"]), 10)
        self.assertGreater(stats["overflow"], 0)
        logits = torch.tensor([[0.0, 10.0, -10.0]])
        self.assertEqual(int(sample_next(logits, top_k=1)), 1)
        self.assertEqual(int(sample_next(logits, top_p=0.1)), 1)
        with self.assertRaises(ValueError):
            sample_next(logits, top_p=0)

    def test_review_requires_approval_and_writes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DraftStore(Path(directory) / "tickets.sqlite3")
            draft = store.create("INC-1", "VPN unavailable", "network", "v1",
                                 "tiny-decoder-demo", "We will review the connection.")
            self.assertEqual(store.ticket_responses("INC-1"), [])
            approved = store.decide(draft["id"], "reviewer-1", True,
                                    "We will check the VPN connection.")
            self.assertEqual(approved["status"], "approved")
            self.assertEqual(len(store.ticket_responses("INC-1")), 1)
            with self.assertRaises(ReviewConflict):
                store.decide(draft["id"], "reviewer-1", True)
            rejected = store.create("INC-2", "Laptop fails", "hardware", "v1",
                                    "tiny-decoder-demo", "We will review the device.")
            store.decide(rejected["id"], "reviewer-2", False, reason="Needs more context")
            self.assertEqual(store.ticket_responses("INC-2"), [])


if __name__ == "__main__":
    unittest.main()
