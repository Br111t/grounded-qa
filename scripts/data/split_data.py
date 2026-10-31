"""Split validated JSONL examples by source group, using a repeatable shuffle."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.data.validate_data import ANSWERABILITY_VALUES, validate_data


def split_examples(
    examples: list[dict], evaluation_groups: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Keep each source together and preserve row order within each split."""
    # Sorting makes group selection independent of the input rows' order.
    source_ids = sorted({example["source_id"] for example in examples})
    if len(source_ids) < 2:
        raise ValueError("At least two source groups are needed for a split.")
    if not 1 <= evaluation_groups < len(source_ids):
        raise ValueError(
            "Evaluation group count must be at least 1 and less than the total "
            f"number of source groups ({len(source_ids)})."
        )

    # A local random generator makes the shuffle repeatable for this seed.
    random.Random(seed).shuffle(source_ids)
    evaluation_ids = set(source_ids[:evaluation_groups])
    training = [row for row in examples if row["source_id"] not in evaluation_ids]
    evaluation = [row for row in examples if row["source_id"] in evaluation_ids]
    return training, evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/source/starter.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/training/train.jsonl"))
    parser.add_argument(
        "--evaluation-output", type=Path, default=Path("data/holdout/evaluation.jsonl")
    )
    parser.add_argument(
        "--evaluation-groups", type=int, default=1,
        help="number of complete source groups reserved for evaluation (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    args = parser.parse_args()

    try:
        paths = (args.input, args.train_output, args.evaluation_output)
        if len({path.resolve() for path in paths}) != 3:
            raise ValueError("Input, training output, and evaluation output must be different files.")

        examples, errors = validate_data(args.input)
        if errors:
            print(f"Input validation failed for {args.input}:")
            for error in errors:
                print(f"  - {error}")
            return 1

        training, evaluation = split_examples(examples, args.evaluation_groups, args.seed)
        outputs = (
            ("Training", args.train_output, training),
            ("Evaluation", args.evaluation_output, evaluation),
        )

        # Check BOTH destinations before writing, to preserve existing work.
        pending_writes = []
        for label, path, rows in outputs:
            if not path.parent.is_dir():
                raise ValueError(f"Output folder does not exist: {path.parent}")
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            content = text.encode("utf-8")
            if path.exists():
                existing = path.read_bytes()
                if existing == content:
                    continue
                if existing:
                    raise ValueError(
                        f"Refusing to replace nonempty output with different data: {path}. "
                        "Choose a new output path to keep the existing file."
                    )
            pending_writes.append((path, content))

        for path, content in pending_writes:
            path.write_bytes(content)

    except (OSError, UnicodeError, ValueError) as error:
        print(f"Split failed: {error}")
        return 1

    print(f"Split ready (seed {args.seed}).")
    for label, path, rows in outputs:
        source_ids = sorted({row["source_id"] for row in rows})
        counts = Counter(row["answerability"] for row in rows)
        print(f"{label}: {len(rows)} examples across {len(source_ids)} source groups -> {path}")
        print("  Sources: " + ", ".join(source_ids))
        print("  " + ", ".join(f"{name}: {counts[name]}" for name in ANSWERABILITY_VALUES))
    print("No source groups overlap. These files are for practicing the workflow.")
    total = len(training) + len(evaluation)

    print(f"Training: {len(training) / total:.1%}")
    print(f"Evaluation: {len(evaluation) / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
