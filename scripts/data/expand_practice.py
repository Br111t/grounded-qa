"""Pair each question in one training source with its equivalent alternate wording."""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re

from scripts.data.generate_data import FIELDS, make_questions
from scripts.data.validate_data import ANSWERABILITY_VALUES, validate_data


TRAIN_PATH = Path("data/training/generated_train.jsonl")
DEFAULT_SOURCE = "generated_workshop_0001"
DEFAULT_OUTPUT = Path("data/training/practice_wordings.jsonl")


def expand_examples(examples: list[dict], source_id: str) -> list[dict]:
    """Keep each original row, then add one variant with the identical target.

    This supports only the generator's known templates. It does not infer new
    facts, paraphrase arbitrary questions, or create answers from model output.
    """
    selected = [row for row in examples if row["source_id"] == source_id]
    if not selected:
        raise ValueError(f"Source ID {source_id!r} was not found in the training data.")

    match = re.match(r"Fictional (Workshop W\d{4,})\. ", selected[0]["information"])
    if match is None:
        raise ValueError("This expansion supports only the fictional workshop format.")
    name = match.group(1)
    requested_fields = [(field,) for field in FIELDS] + list(combinations(FIELDS, 2))
    alternate_questions = {}
    for requested in requested_fields:
        first, second = make_questions(name, requested)
        alternate_questions[first] = second
        alternate_questions[second] = first

    expanded = []
    seen_questions = set()
    used_ids = {row["example_id"] for row in examples}
    for row in selected:
        alternate = alternate_questions.get(row["question"])
        if alternate is None:
            raise ValueError(f"Unknown question format for {row['example_id']!r}; review it manually.")
        if row["question"] in seen_questions or alternate in seen_questions:
            raise ValueError("This source already contains repeated or paired question wordings.")
        seen_questions.update((row["question"], alternate))

        variant_id = row["example_id"] + "_alt"
        if variant_id in used_ids:
            raise ValueError(f"Alternate example ID already exists: {variant_id}")
        used_ids.add(variant_id)
        expanded.append(dict(row))
        # Only the question and example ID change. In particular, keep the
        # source ID so both wordings stay together in any future source split.
        expanded.append({**row, "example_id": variant_id, "question": alternate})
    return expanded


def write_examples(path: Path, examples: list[dict]) -> None:
    """Create a separate JSONL file; an identical rerun leaves it alone."""
    if path.resolve() == TRAIN_PATH.resolve():
        raise ValueError("Choose an output separate from the original training file.")
    if not path.parent.is_dir():
        raise ValueError(f"Output folder does not exist: {path.parent}")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples).encode("utf-8")
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ValueError(f"Output already exists with different data: {path}. Choose a new --output path.")
    with path.open("xb") as output_file:
        output_file.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE, help=f"training source (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="separate review dataset (default: data/training/practice_wordings.jsonl)")
    args = parser.parse_args()

    try:
        examples, errors = validate_data(TRAIN_PATH)
        if errors:
            raise ValueError("Training data validation failed:\n  - " + "\n  - ".join(errors))
        expanded = expand_examples(examples, args.source_id)
        write_examples(args.output, expanded)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Expansion failed: {error}")
        return 1

    counts = Counter(row["answerability"] for row in expanded)
    print(f"Practice dataset ready: {len(expanded)} examples across 1 source group -> {args.output}")
    print(f"Source: {args.source_id}; {len(expanded) // 2} originals, each followed by its _alt wording.")
    print(", ".join(f"{label}: {counts[label]}" for label in ANSWERABILITY_VALUES))
    print("Each pair keeps exactly the same information, source ID, answerability, and reference answer.")
    print("Review the questions and answers before training. No model was loaded or trained.")
    print("After review, use train.py --data to select this file explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
