"""Compare question-sensitive token probabilities for a saved checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from grounded_qa.training.checkpoint import load_checkpoint
from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.tokenization.tokenizer import BOS_ID, EOS_ID, encode


def probability_at_prefix(model, information, question, version, prefix, choices):
    source = torch.tensor([encode(build_prompt(information, question, version=version), add_eos=True)])
    decoder = torch.tensor([[BOS_ID] + encode(prefix)])
    with torch.no_grad():
        probabilities = model(source, decoder)[0, -1].softmax(-1)
    return {
        choice: round(float(probabilities[EOS_ID if choice == "<EOS>" else encode(choice)[0]]), 4)
        for choice in choices
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=Path("data/training/practice_wordings.jsonl")
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, info = load_checkpoint(args.checkpoint)
    version = info.get("prompt_version", "original")
    model.eval()
    rows = {
        row["example_id"]: row
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt version: {version}")
    tests = (
        ("Start", "", ("I", "L")),
        ("After L", "L", ("e", "o")),
        ("After location", "Location: Room Maple.", ("<EOS>", " ")),
    )
    for suffix in ("q01", "q02", "q03", "q08"):
        row = rows[f"generated_workshop_0001_{suffix}"]
        print(f"\n{suffix}: {row['question']}")
        for label, prefix, choices in tests:
            result = probability_at_prefix(
                model, row["information"], row["question"], version, prefix, choices
            )
            print(f"  {label}: {result}")


if __name__ == "__main__":
    main()
