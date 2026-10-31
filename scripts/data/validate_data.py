"""Check JSONL structure and label consistency without changing the data."""

import argparse
import json
from collections import Counter
from pathlib import Path


FALLBACK = "Insufficient information."
REQUIRED_FIELDS = (
    "example_id", "source_id", "information", "question", "answerability", "answer"
)
ANSWERABILITY_VALUES = ("answerable", "partial", "unanswerable")


def validate_data(path: Path) -> tuple[list[dict], list[str]]:
    """Return the parsed examples and readable errors, including line numbers.

    These checks cannot decide whether an answer is supported by the information.
    That still needs a human review, like the review in the exploration notebook.
    """
    examples = []
    errors = []
    seen_ids = {}
    source_information = {}
    information_sources = {}

    # utf-8-sig also accepts a UTF-8 byte-order mark from Windows text editors.
    with path.open(encoding="utf-8-sig") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            location = f"Line {line_number}"
            if not line.strip():
                errors.append(f"{location}: blank lines are not valid examples.")
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{location}: invalid JSON ({error.msg}).")
                continue

            if not isinstance(example, dict):
                errors.append(f"{location}: each line must contain a JSON object.")
                continue

            # Check types before comparing values or using them as dictionary keys.
            invalid_fields = [
                field for field in REQUIRED_FIELDS
                if not isinstance(example.get(field), str) or not example[field].strip()
            ]
            if invalid_fields:
                errors.append(
                    f"{location}: missing or empty/non-string fields: "
                    + ", ".join(invalid_fields) + "."
                )
                continue

            examples.append(example)
            example_id = example["example_id"]
            source_id = example["source_id"]
            information = example["information"]
            answerability = example["answerability"]
            answer = example["answer"]

            if example_id in seen_ids:
                errors.append(
                    f"{location}: duplicate example_id {example_id!r}; "
                    f"first used on line {seen_ids[example_id]}."
                )
            else:
                seen_ids[example_id] = line_number

            # One source ID must always refer to exactly the same information.
            if source_id in source_information:
                if source_information[source_id] != information:
                    errors.append(
                        f"{location}: information differs for source_id {source_id!r}."
                    )
            else:
                source_information[source_id] = information

            # Duplicated source text under different IDs could leak across splits.
            if information in information_sources:
                if information_sources[information] != source_id:
                    errors.append(
                        f"{location}: identical information uses different source IDs: "
                        f"{information_sources[information]!r} and {source_id!r}."
                    )
            else:
                information_sources[information] = source_id

            if answerability not in ANSWERABILITY_VALUES:
                errors.append(
                    f"{location}: answerability must be one of "
                    + ", ".join(ANSWERABILITY_VALUES) + "."
                )
            elif answerability == "unanswerable" and answer != FALLBACK:
                errors.append(f"{location}: unanswerable answer must be exactly {FALLBACK!r}.")
            elif answerability == "partial":
                other_text = answer.replace(FALLBACK, "").strip(" \t\r\n.,:;!?-")
                if FALLBACK not in answer or not other_text:
                    errors.append(
                        f"{location}: partial answer must contain {FALLBACK!r} "
                        "and additional text for the supported part."
                    )
            elif answerability == "answerable" and FALLBACK.casefold() in answer.casefold():
                errors.append(f"{location}: answerable answer must not include the fallback.")

    if not examples:
        errors.append("File contains no examples with all required string fields.")
    return examples, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("data/source/starter.jsonl"),
        help="JSONL file to check, relative to the current folder (default: data/source/starter.jsonl)",
    )
    args = parser.parse_args()

    try:
        examples, errors = validate_data(args.path)
    except (OSError, UnicodeError) as error:
        print(f"Could not read {args.path}: {error}")
        return 1

    if errors:
        print(f"Validation failed for {args.path}:")
        for error in errors:
            print(f"  - {error}")
        return 1

    source_count = len({example["source_id"] for example in examples})
    counts = Counter(example["answerability"] for example in examples)
    print(f"Validation passed: {len(examples)} examples across {source_count} source groups.")
    print(", ".join(f"{label}: {counts[label]}" for label in ANSWERABILITY_VALUES))
    print("Structure and label wording checked; factual support still requires human review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
