"""Train both synthetic demo stages for the end-to-end local workflow."""

from pathlib import Path
import json

from src.llm.inference.train_demo import train_demo
from src.training.train_neural import train


def main() -> None:
    classifier = Path("models/transformer_demo.json")
    generator = Path("models/llm_demo.pt")
    settings = json.loads(Path("configs/llm/demo.json").read_text(encoding="utf-8"))
    print(train(Path("data/samples/applications/all_applications.csv"),
                Path("configs/application_sample_split.json"), classifier, "v5"))
    print(train_demo(generator, Path(settings["dataset"]), settings["steps"],
                     settings["seed"], settings["model"]))


if __name__ == "__main__":
    main()
