"""Evidence Sensitivity V3: diverse values, fixed question, segregated development tests.

Run from the repository root:
    python src/generate_evidence_data_v3.py --seed 42

No V1/V2 dataset or original holdout is overwritten.
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path

QUESTION = "What is the leader of Workshop W0001?"
FALLBACK = "Insufficient information."

# Distinguish an unseen FULL name made from known syllables ("recombined")
# from a name using syllables absent from all training leader values ("unseen").
TRAIN_START = ("Ma", "Ta", "Sa", "Ni", "Zu", "Le", "Ve", "Ki", "Ra", "Jo", "Pa", "No")
TRAIN_END = ("ra", "vi", "na", "ko", "mi", "la", "ri", "to")
UNSEEN_START = ("Xe", "Qu", "Fy", "Wo", "Dy", "Bo")
UNSEEN_END = ("zen", "vor", "pex", "mir")

ROOMS = ("Room Maple", "Room Cedar", "Room Birch", "Room Pine", "Room Elm", "Room Ash")
TOPICS = (None, "Python", "Gardening", "Astronomy")
CONTEXTS = tuple(product(ROOMS, (False, True), TOPICS))

OUTPUTS = {
    "train": Path("experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_train.jsonl"),
    "known": Path("experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_known.jsonl"),
    "recombined": Path("experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_recombined.jsonl"),
    "unseen": Path("experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_unseen.jsonl"),
    "missing": Path("experiments/evidence_sensitivity_v3/data/evidence_sensitivity_v3_missing.jsonl"),
}


def assemble(starts: tuple[str, ...], ends: tuple[str, ...]) -> list[str]:
    return [start + end for start, end in product(starts, ends)]


def information_for(leader: str | None, context: tuple[str, bool, str | None]) -> str:
    room, reversed_order, topic = context
    facts = [f"Location: {room}."]
    if leader is not None:
        facts.append(f"Leader: {leader}.")
    if topic is not None:
        facts.append(f"Topic: {topic}.")
    if reversed_order:
        facts.reverse()
    return "Fictional Workshop W0001. " + " ".join(facts)


def make_row(group: str, index: int, leader: str | None, context: tuple) -> dict:
    source_id = f"evidence_v3_{group}_{index:04d}"
    return {
        "example_id": f"{source_id}_q01",
        "source_id": source_id,
        "information": information_for(leader, context),
        "question": QUESTION,
        "answerability": "answerable" if leader is not None else "unanswerable",
        "answer": f"Leader: {leader}." if leader is not None else FALLBACK,
    }


def build_datasets(seed: int = 42) -> dict[str, list[dict]]:
    """168 train; 48 known, 24 recombined, 24 unseen, 12 missing eval."""
    rng = random.Random(seed)
    candidate_names = assemble(TRAIN_START, TRAIN_END)
    rng.shuffle(candidate_names)
    train_names = candidate_names[:48]
    recombined_names = candidate_names[48:60]

    novel_names = assemble(UNSEEN_START, UNSEEN_END)
    rng.shuffle(novel_names)
    unseen_names = novel_names[:12]

    if set(train_names) & (set(recombined_names) | set(unseen_names)):
        raise ValueError("Training leader names must be disjoint from both novel categories.")
    if set(recombined_names) & set(unseen_names):
        raise ValueError("Recombined and unseen names must not overlap.")

    entries = {group: [] for group in OUTPUTS}
    for name in train_names:
        contexts = rng.sample(CONTEXTS, 4)
        entries["train"].extend((name, context) for context in contexts[:3])
        entries["known"].append((name, contexts[3]))

    for name in recombined_names:
        entries["recombined"].extend(
            (name, context) for context in rng.sample(CONTEXTS, 2)
        )

    for name in unseen_names:
        entries["unseen"].extend(
            (name, context) for context in rng.sample(CONTEXTS, 2)
        )

    # When the only field is Location, reversing its order does nothing.
    # Exclude that duplicate context from the missing-evidence sampling pool.
    unique_missing_contexts = [
        context for context in CONTEXTS
        if context[2] is not None or not context[1]
    ]
    missing_contexts = rng.sample(unique_missing_contexts, 36)
    entries["train"].extend((None, context) for context in missing_contexts[:24])
    entries["missing"].extend((None, context) for context in missing_contexts[24:])

    # Same seeded, deterministic shuffle; avoid long same-name update runs.
    for group in entries:
        rng.shuffle(entries[group])

    datasets = {
        group: [
            make_row(group, index, name, context)
            for index, (name, context) in enumerate(pairs, start=1)
        ]
        for group, pairs in entries.items()
    }
    expected = {"train": 168, "known": 48, "recombined": 24,
                "unseen": 24, "missing": 12}
    if {key: len(rows) for key, rows in datasets.items()} != expected:
        raise ValueError("Unexpected partition size.")

    all_rows = [row for rows in datasets.values() for row in rows]
    if len({row["information"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate evidence text across data partitions.")
    if len({row["source_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate source IDs across data partitions.")
    if any(row["question"] != QUESTION for row in all_rows):
        raise ValueError("The question must stay fixed.")
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = build_datasets(args.seed)

    # Fail safely if a different version already exists, before writing.
    pending = []
    for group, path in OUTPUTS.items():
        content = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in datasets[group]
        ).encode("utf-8")
        if path.exists():
            if path.read_bytes() == content:
                continue  # Re-running the same seed is safe.
            raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
        pending.append((path, content))
    for path, content in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    for group, rows in datasets.items():
        print(f"{group}: {len(rows)} examples -> {OUTPUTS[group]}")
    print("Fixed question; leader names split into distinct known/recombined/unseen groups.")
    print("Evaluation files are development sets. Original holdout untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
