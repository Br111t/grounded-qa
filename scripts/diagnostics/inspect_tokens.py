"""Report byte-token lengths without truncating text or changing dataset files."""

import argparse
from pathlib import Path
from statistics import mean

from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.tokenization.tokenizer import encode
from scripts.data.validate_data import validate_data


def measure_examples(examples: list[dict]) -> list[dict]:
    """Count the complete prompt and answer, each with one EOS and no BOS."""
    measurements = []
    for example in examples:
        prompt = build_prompt(example["information"], example["question"])
        measurements.append({
            "example_id": example["example_id"],
            "input_tokens": len(encode(prompt, add_eos=True)),
            "target_tokens": len(encode(example["answer"], add_eos=True)),
        })
    return measurements


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("token limits must be at least 1")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path,
        default=[Path("data/training/generated_train.jsonl"), Path("data/holdout/generated_evaluation.jsonl")],
        help="datasets to inspect (default: both generated split files)",
    )
    parser.add_argument("--max-input-tokens", type=positive_int, help="optional prompt length limit")
    parser.add_argument("--max-target-tokens", type=positive_int, help="optional answer length limit")
    args = parser.parse_args()

    print("Counts include one EOS per input/target, no BOS, and no padding.")
    failed = False
    for path in args.paths:
        try:
            examples, errors = validate_data(path)
            if errors:
                print(f"\nValidation failed for {path}:")
                for error in errors:
                    print(f"  - {error}")
                failed = True
                continue
            measurements = measure_examples(examples)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"Could not inspect {path}: {error}")
            failed = True
            continue

        print(f"\n{path}: {len(examples)} examples")
        for label, field, limit in (
            ("Input", "input_tokens", args.max_input_tokens),
            ("Target", "target_tokens", args.max_target_tokens),
        ):
            lengths = [row[field] for row in measurements]
            longest = max(measurements, key=lambda row: row[field])
            print(f"  {label}: min={min(lengths)}, mean={mean(lengths):.1f}, max={max(lengths)}")
            print(f"    First longest example: {longest['example_id']}")
            if limit is not None:
                violations = [row for row in measurements if row[field] > limit]
                if violations:
                    failed = True
                    for row in violations:
                        print(f"    OVER LIMIT: {row['example_id']} has {row[field]} {label.lower()} tokens; limit={limit}")
                else:
                    print(f"    All examples fit the {limit}-token {label.lower()} limit.")

    if failed:
        print("\nInspection failed. Resolve the reported issues; no text was truncated.")
        return 1
    print("\nInspection passed. No files were changed and no text was truncated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
