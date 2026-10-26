"""Generate fictional workshop QA examples with deliberately missing facts."""

import argparse
import json
import random
from collections import Counter
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

from validate_data import ANSWERABILITY_VALUES, FALLBACK


FIELDS = ("date", "location", "leader", "seats")
LABELS = {"date": "Date", "location": "Location", "leader": "Leader", "seats": "Seats"}
QUESTION_NAMES = {"date": "date", "location": "location", "leader": "leader", "seats": "seat count"}
ROOMS = ("Room Cedar", "Room Maple", "Room Birch", "Room Pine", "Room Elm", "Room Ash")
LEADERS = ("Ari", "Mira", "Niko", "Sana", "Tavi", "Zuri")

QUESTION_TEMPLATES = {
    ("date",): (
        "What is the date of {name}?",
        "When is {name} scheduled?",
    ),
    ("location",): (
        "For {name}, give the location.",
        "Where is {name} located?",
    ),
    ("leader",): (
        "Who is the leader of {name}?",
        "For {name}, give the leader's name.",
    ),
    ("seats",): (
        "What is the seat count of {name}?",
        "How many seats does {name} have?",
    ),
    ("date", "location"): (
        "What are the date and location of {name}?",
        "When and where will {name} take place?",
    ),
    ("date", "leader"): (
        "What is the date of {name}, and who is its leader?",
        "For {name}, give the date and leader's name.",
    ),
    ("date", "seats"): (
        "What are the date and seat count of {name}?",
        "When is {name}, and how many seats does it have?",
    ),
    ("location", "leader"): (
        "Where is {name} located, and who is its leader?",
        "For {name}, give the location and leader's name.",
    ),
    ("location", "seats"): (
        "What are the location and seat count of {name}?",
        "Where is {name}, and how many seats does it have?",
    ),
    ("leader", "seats"): (
        "For {name}, give the leader's name and seat count.",
        "Who leads {name}, and how many seats does it have?",
    ),
}


def make_questions(
    name: str,
    requested: tuple[str, ...],
) -> tuple[str, str]:
    """Return two natural phrasings requesting the same facts."""

    try:
        first, second = QUESTION_TEMPLATES[requested]
    except KeyError as error:
        raise ValueError(
            f"Unsupported field combination: {requested}"
        ) from error

    return first.format(name=name), second.format(name=name)


def make_answer(requested: tuple[str, ...], visible: dict[str, str]) -> tuple[str, str]:
    """Derive the label and target using only facts included in the information."""
    supported = [field for field in requested if field in visible]
    missing = [field for field in requested if field not in visible]
    if not supported:
        return "unanswerable", FALLBACK

    # Supported parts come first, even if a missing part was asked about first.
    parts = [f"{LABELS[field]}: {visible[field]}." for field in supported]
    parts.extend(f"{LABELS[field]}: {FALLBACK}" for field in missing)
    answerability = "partial" if missing else "answerable"
    return answerability, " ".join(parts)


def generate_examples(records: int = 12, seed: int = 42) -> list[dict]:
    """Make ten related questions per source with a repeatable local generator."""
    if records < 1:
        raise ValueError("records must be at least 1.")

    rng = random.Random(seed)
    visibility_patterns = list(combinations(FIELDS, 2))
    question_fields = [(field,) for field in FIELDS] + list(combinations(FIELDS, 2))
    examples = []

    for index in range(records):
        # Every six sources cover all six possible pairs of visible fields.
        if index % len(visibility_patterns) == 0:
            rng.shuffle(visibility_patterns)
        visible_fields = visibility_patterns[index % len(visibility_patterns)]
        source_id = f"generated_workshop_{index + 1:04d}"
        name = f"Workshop W{index + 1:04d}"

        # These values are fictional. Hidden values must never enter an answer.
        full_record = {
            "date": (date(2030, 1, 1) + timedelta(days=rng.randrange(365))).isoformat(),
            "location": rng.choice(ROOMS),
            "leader": rng.choice(LEADERS),
            "seats": str(rng.randrange(4, 33, 2)),
        }
        visible = {field: full_record[field] for field in visible_fields}
        displayed_fields = list(visible)
        rng.shuffle(displayed_fields)
        facts = [f"{LABELS[field]}: {visible[field]}." for field in displayed_fields]
        information = f"Fictional {name}. " + " ".join(facts)

        for question_index, requested in enumerate(question_fields, start=1):
            # Choose one phrasing per question, independently of its answerability.
            question = rng.choice(make_questions(name, requested))
            answerability, answer = make_answer(requested, visible)
            examples.append({
                "example_id": f"{source_id}_q{question_index:02d}",
                "source_id": source_id,
                "information": information,
                "question": question,
                "answerability": answerability,
                "answer": answer,
            })

    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=12, help="source records, not question count (default: 12)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=Path("data/source/generated.jsonl"))
    args = parser.parse_args()

    try:
        examples = generate_examples(args.records, args.seed)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples)
        content = text.encode("utf-8")
        if not args.output.parent.is_dir():
            raise ValueError(f"Output folder does not exist: {args.output.parent}")
        existing = args.output.read_bytes() if args.output.exists() else b""
        if existing and existing != content:
            raise ValueError(
                f"Refusing to replace nonempty output with different data: {args.output}. "
                "Use --output to choose a new file."
            )
        if existing != content:
            args.output.write_bytes(content)
    except (OSError, ValueError) as error:
        print(f"Generation failed: {error}")
        return 1

    counts = Counter(row["answerability"] for row in examples)
    print(f"Dataset ready: {len(examples)} examples across {args.records} source groups -> {args.output}")
    print(", ".join(f"{label}: {counts[label]}" for label in ANSWERABILITY_VALUES))
    print(f"Seed: {args.seed}. The generated data has not been split or used for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
