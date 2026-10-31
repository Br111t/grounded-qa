
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

from grounded_qa.training.checkpoint import load_checkpoint
from grounded_qa.inference.predict import generate_answer, matches_reference
from scripts.data.validate_data import validate_data

torch.set_num_threads(1)

checkpoint = Path("models/evidence_sensitivity_v2b_2000_steps.pt")
dataset = Path("data/training/evidence_sensitivity_v2_train.jsonl")

model, info = load_checkpoint(checkpoint)
rows, errors = validate_data(dataset)

if errors:
    raise ValueError(errors)

for label in ("answerable", "unanswerable"):
    examples = [r for r in rows if r["answerability"] == label]
    correct = 0

    for row in examples:
        result = generate_answer(
            model,
            row["information"],
            row["question"],
            prompt_version=info.get("prompt_version", "original"),
        )
        correct += int(matches_reference(result, row["answer"]))

    print(f"{label}: {correct}/{len(examples)} "
          f"({correct / len(examples):.1%})")
