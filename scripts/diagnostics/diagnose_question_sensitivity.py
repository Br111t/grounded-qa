
import json
from pathlib import Path

import torch

from grounded_qa.training.checkpoint import load_checkpoint
from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.tokenization.tokenizer import BOS_ID, EOS_ID, encode

torch.set_num_threads(1)

model, _ = load_checkpoint(
    Path("models/workshop_0001_wordings_4000_steps.pt")
)
model.eval()

path = Path("data/training/practice_wordings.jsonl")
rows = {
    row["example_id"]: row
    for row in (
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )
}

def inspect(row, prefix, choices):
    source = torch.tensor([
        encode(build_prompt(row["information"], row["question"]),
               add_eos=True)
    ])
    decoder = torch.tensor([[BOS_ID] + encode(prefix)])

    with torch.no_grad():
        probabilities = model(source, decoder)[0, -1].softmax(-1)

    return {
        choice: round(float(probabilities[
            EOS_ID if choice == "<EOS>" else encode(choice)[0]
        ]), 4)
        for choice in choices
    }

tests = [
    ("Start of answer", "", ("I", "L")),
    ("After L", "L", ("e", "o")),
    ("After location", "Location: Room Maple.", ("<EOS>", " ")),
]

for suffix in ("q01", "q02", "q03", "q08"):
    row = rows[f"generated_workshop_0001_{suffix}"]
    print(f"\n{suffix}: {row['question']}")

    for label, prefix, choices in tests:
        print(f"  {label}: {inspect(row, prefix, choices)}")
