"""Generate a controlled evidence-sensitivity experiment without modifying V1.

The question is identical for all rows. Only the information varies.
Separate train, known-name, unseen-name and missing-evidence sets are written.
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path

QUESTION = "What is the leader of Workshop W0001?"
TRAIN_NAMES = (
    "Mira", "Tavi", "Sana", "Ari", "Niko", "Zuri",
    "Lena", "Omar", "Vera", "Imani", "Kira", "Pablo",
)
UNSEEN_NAMES = ("Elio", "Quinn", "Becca", "Yara")
ROOMS = ("Room Maple", "Room Cedar", "Room Birch", "Room Pine", "Room Elm", "Room Ash")
TOPICS = (None, "Python", "Gardening", "Astronomy")
CONTEXTS = list(product(ROOMS, (False, True), TOPICS))

OUTPUTS = {
    "train": Path("data/training/evidence_sensitivity_v2_train.jsonl"),
    "known": Path("data/evaluation/evidence_sensitivity_v2_known.jsonl"),
    "unseen": Path("data/evaluation/evidence_sensitivity_v2_unseen.jsonl"),
    "missing": Path("data/evaluation/evidence_sensitivity_v2_missing.jsonl"),
}


def information_for(leader: str | None, context: tuple[str, bool, str | None]) -> str:
    room, reverse_order, topic = context
    facts = [f"Location: {room}."]
    if leader is not None:
        facts.append(f"Leader: {leader}.")
    if topic is not None:
        facts.append(f"Topic: {topic}.")
    if reverse_order:
        facts.reverse()
    return "Fictional Workshop W0001. " + " ".join(facts)


def make_row(group: str, index: int, leader: str | None, context: tuple) -> dict:
    source_id = f"evidence_v2_{group}_{index:04d}"
    return {
        "example_id": f"{source_id}_q01",
        "source_id": source_id,
        "information": information_for(leader, context),
        "question": QUESTION,
        "answerability": "answerable" if leader is not None else "unanswerable",
        "answer": f"Leader: {leader}." if leader is not None else "Insufficient information.",
    }


def build_datasets(seed: int = 42) -> dict[str, list[dict]]:
    """Build 108 training + 24 known + 24 unseen + 12 missing evaluation rows."""
    if not set(TRAIN_NAMES).isdisjoint(UNSEEN_NAMES):
        raise ValueError("Unseen leader names must not appear in training.")
    rng = random.Random(seed)
    entries: dict[str, list[tuple[str | None, tuple]]] = {
        key: [] for key in OUTPUTS
    }

    for name in TRAIN_NAMES:
        contexts = rng.sample(CONTEXTS, 10)
        entries["train"].extend((name, context) for context in contexts[:8])
        entries["known"].extend((name, context) for context in contexts[8:])

    for name in UNSEEN_NAMES:
        entries["unseen"].extend(
            (name, context) for context in rng.sample(CONTEXTS, 6)
        )

    # Missing-leader training and evaluation contexts cannot overlap.
    missing_contexts = rng.sample(CONTEXTS, 24)
    entries["train"].extend((None, context) for context in missing_contexts[:12])
    entries["missing"].extend((None, context) for context in missing_contexts[12:])

    # Shuffle rows deterministically so training does not repeatedly see an
    # uninterrupted run of examples sharing the same leader.
    for group in entries:
        rng.shuffle(entries[group])

    datasets = {
        group: [
            make_row(group, i, name, context)
            for i, (name, context) in enumerate(pairs, start=1)
        ]
        for group, pairs in entries.items()
    }
    expected_counts = {"train": 108, "known": 24, "unseen": 24, "missing": 12}
    for group, count in expected_counts.items():
        if len(datasets[group]) != count:
            raise ValueError(f"{group}: expected {count} rows.")
    all_rows = [row for rows in datasets.values() for row in rows]
    if len({row["information"] for row in all_rows}) != len(all_rows):
        raise ValueError("Different partitions contain duplicate information.")
    if len({row["source_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Source IDs must be unique across partitions.")
    if any(row["question"] != QUESTION for row in all_rows):
        raise ValueError("Every question must remain identical.")
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = build_datasets(args.seed)

    # Check all destinations before writing any of them.
    pending = []
    for group, path in OUTPUTS.items():
        content = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in datasets[group]
        ).encode("utf-8")
        if path.exists():
            existing = path.read_bytes()
            if existing == content:
                continue
            if existing:
                raise FileExistsError(
                    f"Refusing to overwrite {path}; choose a fresh dataset version."
                )
        pending.append((path, content))

    for path, content in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    for group, rows in datasets.items():
        print(f"{group}: {len(rows)} examples -> {OUTPUTS[group]}")
    print(f"Training leader names: {', '.join(TRAIN_NAMES)}")
    print(f"Unseen evaluation names: {', '.join(UNSEEN_NAMES)}")
    print("Fixed question across all partitions; distinct input strings.")
    print("All evaluation files are development sets; existing final holdout untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
