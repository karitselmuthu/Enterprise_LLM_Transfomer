"""Incremental decoding and local synthetic-checkpoint response drafting."""

from pathlib import Path

import torch

from src.llm.architectures.decoder import DecoderConfig, TinyDecoderLM
from src.llm.inference.tokenizer import DemoTokenizer
from src.llm.sampling.strategies import sample_next


def generate_tokens(model: TinyDecoderLM, prompt: list[int], max_new_tokens: int = 48,
                    eos_id: int | None = None, temperature: float = 0,
                    top_k: int | None = None, top_p: float | None = None,
                    seed: int = 42, use_cache: bool = True) -> list[int]:
    if not prompt or max_new_tokens < 1:
        raise ValueError("Generation needs a prompt and positive token limit")
    if len(prompt) >= model.config.max_context:
        raise ValueError("Prompt leaves no context for generation")
    device = next(model.parameters()).device
    random_generator = torch.Generator(device=device).manual_seed(seed)
    result = list(prompt)
    cache = None
    model.eval()
    with torch.inference_mode():
        for _ in range(min(max_new_tokens, model.config.max_context - len(prompt))):
            current = [result[-1]] if use_cache and cache is not None else result
            logits, new_cache = model(torch.tensor([current], dtype=torch.long, device=device),
                                      cache if use_cache else None)
            if use_cache:
                cache = new_cache
            chosen = int(sample_next(logits[:, -1, :], temperature, top_k, top_p,
                                     random_generator)[0, 0])
            if chosen == eos_id:
                break
            result.append(chosen)
    return result[len(prompt):]


class DemoLLMGenerator:
    """Educational local decoder; generated text always needs human review."""

    def __init__(self, checkpoint: Path):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.tokenizer = DemoTokenizer(payload["vocabulary"])
        self.model = TinyDecoderLM(DecoderConfig(**payload["config"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.version = "tiny-decoder-demo"

    def draft(self, ticket_text: str, label: str) -> str:
        vocab = self.tokenizer.vocabulary
        prompt = ([vocab["<bos>"]] + self.tokenizer.encode(label) +
                  self.tokenizer.encode(ticket_text)[:12] + [vocab["<sep>"]])
        generated = generate_tokens(self.model, prompt, eos_id=vocab["<eos>"],
                                    max_new_tokens=48, temperature=0)
        response = self.tokenizer.decode(generated)
        if not response:
            raise RuntimeError("Demo language model produced an empty draft")
        return response
