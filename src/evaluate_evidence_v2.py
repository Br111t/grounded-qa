"""Evaluate V2 evidence sensitivity by known names, unseen names, and missing evidence."""

import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from evidence_v2_validation import validate_v2_rows
from generate_evidence_data_v2 import OUTPUTS
from predict import generate_answer, matches_reference
from validate_data import validate_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--group", choices=("known", "unseen", "missing", "all"), default="all",
        help="development evaluation category (default: all)",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    model, training_info = load_checkpoint(args.checkpoint)
    prompt_version = training_info.get("prompt_version", "original")
    if training_info.get("training_file") != OUTPUTS["train"].as_posix():
        raise ValueError(
            "This evaluator requires a checkpoint trained on the registered V2 training file."
        )
    groups = (
        ("known", "unseen", "missing")
        if args.group == "all" else (args.group,)
    )
    print(f"Checkpoint: {args.checkpoint}; prompt: {prompt_version}")
    print("These are development evaluations, not final holdout results.")
    total_correct = 0
    total_count = 0

    for group in groups:
        path = OUTPUTS[group]
        rows, errors = validate_data(path)
        if errors:
            raise ValueError(f"Invalid {group} evaluation data: {'; '.join(errors)}")
        validate_v2_rows(rows, group=group)
        correct = 0
        failures = []
        for row in rows:
            result = generate_answer(
                model, row["information"], row["question"],
                prompt_version=prompt_version,
            )
            match = matches_reference(result, row["answer"])
            correct += int(match)
            if not match:
                failures.append((row["example_id"], result.text, row["answer"], result.ended_with_eos))
        count = len(rows)
        total_correct += correct
        total_count += count
        print(f"\n{group}: {correct}/{count} exact matches ({correct / count:.1%})")
        for example_id, prediction, reference, ended in failures[:5]:
            print(f"  {example_id}: generated={prediction!r}; expected={reference!r}; EOS={ended}")
        if len(failures) > 5:
            print(f"  ... {len(failures) - 5} other mismatches omitted")

    if len(groups) > 1:
        print(f"\nOverall development: {total_correct}/{total_count} ({total_correct / total_count:.1%})")
    print("No weights or datasets changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
