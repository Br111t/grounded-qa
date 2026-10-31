"""Print model inputs and separate reference answers without running a model."""

import argparse
from pathlib import Path

from grounded_qa.prompting.prompting import build_prompt
from scripts.data.validate_data import validate_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/training/train.jsonl"),
        help="JSONL dataset to preview (default: data/training/train.jsonl)",
    )
    parser.add_argument(
        "--example-id",
        help="preview one example by ID; otherwise show the first example of each type",
    )
    args = parser.parse_args()

    try:
        examples, errors = validate_data(args.data)
    except (OSError, UnicodeError) as error:
        print(f"Could not read {args.data}: {error}")
        return 1

    if errors:
        print(f"Input validation failed for {args.data}:")
        for error in errors:
            print(f"  - {error}")
        return 1

    if args.example_id is not None:
        selected = [row for row in examples if row["example_id"] == args.example_id]
        if not selected:
            print(f"Example ID {args.example_id!r} was not found in {args.data}.")
            return 1
    else:
        selected = []
        seen_types = set()
        for example in examples:
            if example["answerability"] not in seen_types:
                selected.append(example)
                seen_types.add(example["answerability"])

    print(f"Previewing {len(selected)} examples from {args.data}.")
    for example in selected:
        # IDs and labels are displayed for our review, outside the model input.
        print(f"\nExample: {example['example_id']} ({example['answerability']})")
        print("--- MODEL INPUT ---")
        print(build_prompt(example["information"], example["question"]))
        print("--- EXPECTED ANSWER (dataset reference, not model output) ---")
        print(example["answer"])

    print("\nPreview complete. No model was loaded or run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
