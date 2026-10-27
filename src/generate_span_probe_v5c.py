"""Generate V5C read-only 2x2 span-boundary diagnostic (NOT training data).

Factors: 4 vs 5 ASCII characters; vowel vs consonant ending. Each of
12 matched name stems appears in all four groups, in identical contexts.
No original V3/V5B dataset is changed, and no probe data is trainable.
"""

import argparse
import json
import random
from pathlib import Path

from evidence_v3_validation import parse_fields
from generate_evidence_data_v3 import (
    CONTEXTS, TRAIN_START, build_datasets, information_for,
)
from generate_span_data_v5b import build_additions
from validate_data import validate_data

SEED = 42
QUESTION = "What is the leader of Workshop W0001?"
GROUPS = (
    "length4_vowel", "length4_consonant",
    "length5_vowel", "length5_consonant",
)
PATHS = {
    group: Path(f"data/evaluation/evidence_span_v5c_{group}.jsonl")
    for group in GROUPS
}
ROWS_PER_GROUP = 24
STEMS = 12
CONTEXTS_PER_STEM = 2
# 6 stems per final-character pair. Vowels and consonants are varied
# within each length instead of using a single terminal character.
ENDING_PAIRS = (("a", "n"), ("i", "t"))
MIDDLE_INSERTION = "e"  # same character for BOTH 5-character ending types
THIRD_LETTERS = ("v", "r", "l", "m", "k", "s", "p")


def _leader_names(groups: dict[str, list[dict]]) -> set[str]:
    return {
        parse_fields(row["information"], row["source_id"])["Leader"]
        for rows in groups.values()
        for row in rows
        if row["answerability"] == "answerable"
    }


def build_probe_datasets(seed: int = SEED) -> dict[str, list[dict]]:
    """Create fixed development-only pairs with disjoint leader names."""
    if seed != SEED:
        raise ValueError("V5C diagnostic is locked to seed 42.")
    reserved_names = (
        _leader_names(build_datasets(seed=seed))
        | _leader_names(build_additions(seed=seed))
    )
    rng = random.Random(seed)
    rows = {group: [] for group in GROUPS}
    selected_names: set[str] = set()
    starters = list(TRAIN_START)
    rng.shuffle(starters)
    if len(starters) != STEMS:
        raise ValueError("Unexpected V3 start-syllable definitions.")

    for pair_index, start_syllable in enumerate(starters):
        vowel, consonant = ENDING_PAIRS[pair_index % len(ENDING_PAIRS)]
        third_options = list(THIRD_LETTERS)
        rng.shuffle(third_options)

        chosen = None
        for third in third_options:
            stem = start_syllable + third
            proposals = {
                "length4_vowel": stem + vowel,
                "length4_consonant": stem + consonant,
                "length5_vowel": stem + MIDDLE_INSERTION + vowel,
                "length5_consonant": stem + MIDDLE_INSERTION + consonant,
            }
            all_names = set(proposals.values())
            if (
                len(all_names) == len(GROUPS)
                and not (all_names & reserved_names)
                and not (all_names & selected_names)
            ):
                chosen = proposals
                break
        if chosen is None:
            raise ValueError(f"Could not find a disjoint diagnostic stem for {start_syllable}.")
        selected_names.update(chosen.values())

        # Compare matching row indices across groups: same workshop context,
        # same field ordering, same topic, only name length/ending differ.
        contexts = rng.sample(CONTEXTS, CONTEXTS_PER_STEM)
        for context in contexts:
            for group in GROUPS:
                name = chosen[group]
                index = len(rows[group]) + 1
                source_id = f"evidence_span_v5c_{group}_{index:04d}"
                rows[group].append({
                    "example_id": f"{source_id}_q01",
                    "source_id": source_id,
                    "information": information_for(name, context),
                    "question": QUESTION,
                    "answerability": "answerable",
                    "answer": f"Leader: {name}.",
                })
    if any(len(part) != ROWS_PER_GROUP for part in rows.values()):
        raise ValueError("Unexpected 2x2 diagnostic partition size.")
    return rows


def validate_probe_group(group: str, rows: list[dict]) -> None:
    """Lock every diagnostic reference, ordering and context to seed 42."""
    if group not in GROUPS:
        raise ValueError(f"Unsupported V5C diagnostic group: {group}")
    if rows != build_probe_datasets()[group]:
        raise ValueError(f"V5C group {group} differs from its locked seed-42 rows.")
    required_length = 4 if group.startswith("length4") else 5
    expected_vowel = group.endswith("_vowel")
    for row in rows:
        leader = parse_fields(row["information"], row["source_id"]).get("Leader")
        if (
            not leader or not leader.isascii() or not leader.isalpha()
            or len(leader) != required_length
            or (leader[-1].lower() in "aeiou") != expected_vowel
            or row["answer"] != f"Leader: {leader}."
        ):
            raise ValueError(f"Invalid diagnostic name or reference: {row['example_id']}")


def read_probe_group(group: str) -> list[dict]:
    if group not in PATHS:
        raise ValueError(f"Unsupported V5C group: {group}")
    rows, errors = validate_data(PATHS[group])
    if errors:
        raise ValueError("V5C structural validation failed: " + "; ".join(errors))
    validate_probe_group(group, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("V5C uses locked seed 42 only.")
    groups = build_probe_datasets()
    pending: list[tuple[Path, bytes]] = []
    # First verify every output. Do not partially write if any file conflicts.
    for group, path in PATHS.items():
        validate_probe_group(group, groups[group])
        encoded = "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in groups[group]
        ).encode("utf-8")
        if path.exists():
            if path.read_bytes() != encoded:
                raise FileExistsError(f"Refusing to overwrite modified V5C data: {path}")
        else:
            pending.append((path, encoded))
    for path, encoded in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(encoded)
    for group in GROUPS:
        print(f"{group}: {len(groups[group])} development-only rows -> {PATHS[group]}")
    print("All four groups share matched contexts. V3/V5B train and development files unchanged.")
    print("No training data generated; no final holdout read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
