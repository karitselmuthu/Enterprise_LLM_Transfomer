"""Train an offline toy decoder on fictional support-response examples."""

import argparse
import json
from pathlib import Path

import torch

from src.llm.architectures.decoder import DecoderConfig, TinyDecoderLM
from src.llm.inference.tokenizer import DemoTokenizer
from src.training.train import load_rows


REPLIES = {
    "access": "Thanks for reporting the access issue. A support agent will verify your account and follow up with approved recovery steps.",
    "hardware": "Thanks for reporting the hardware issue. A support agent will review the device details and arrange the appropriate next step.",
    "network": "Thanks for reporting the network issue. A support agent will check the connection details and follow up with next steps.",
    "software": "Thanks for reporting the software issue. A support agent will review the error details and follow up with next steps.",
}


def train_demo(output: Path, data: Path = Path("data/samples/tickets.csv"),
               steps: int = 150, seed: int = 42, model_settings: dict | None = None) -> dict:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    rows = load_rows(data)
    if set(row["label"] for row in rows) != set(REPLIES):
        raise ValueError("Demo training data must use the four synthetic class labels")
    tokenizer = DemoTokenizer.build([row["text"] for row in rows] +
                                    list(REPLIES) + list(REPLIES.values()))
    vocab = tokenizer.vocabulary
    sequences = [[vocab["<bos>"]] + tokenizer.encode(row["label"]) +
                 tokenizer.encode(row["text"])[:12] + [vocab["<sep>"]] +
                 tokenizer.encode(REPLIES[row["label"]]) + [vocab["<eos>"]]
                 for row in rows]
    length = max(map(len, sequences))
    tokens = torch.tensor([sequence + [vocab["<pad>"]] * (length - len(sequence))
                           for sequence in sequences], dtype=torch.long)
    settings = model_settings or {"dimension": 32, "heads": 4, "kv_heads": 2, "layers": 1,
                                  "max_context": 64, "experts": 0}
    config = DecoderConfig(vocab_size=len(vocab),
                           **{**settings, "max_context": max(settings["max_context"], length + 4)})
    model = TinyDecoderLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model.next_token_loss(tokens)
        loss.backward()
        optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "vocabulary": vocab,
                "state_dict": model.state_dict()}, output)
    return {"steps": steps, "examples": len(rows),
            "final_loss": round(float(loss.detach()), 4), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/llm_demo.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/samples/tickets.csv"))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--config", type=Path, help="Optional configs/llm/demo.json settings")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    data = Path(settings.get("dataset", args.data))
    print(train_demo(args.output, data, settings.get("steps", args.steps),
                     settings.get("seed", 42), settings.get("model")))


if __name__ == "__main__":
    main()
