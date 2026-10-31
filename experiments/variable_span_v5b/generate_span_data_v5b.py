"""Generate strictly separated V5B additional training and length-control dev JSONL.

Existing V3 files remain unchanged. V5B trains on V3 train plus 72 extra
examples: three presentations each of 8 distinct names at lengths 3/5/6.
The separate dev groups are familiar-pattern length 5 and unfamiliar-pattern
length 4; V3 recombined (4) and V3 unseen (5) complete the comparison.
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path

from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import (
    CONTEXTS, TRAIN_END, TRAIN_START, UNSEEN_START, information_for,
)
from scripts.data.validate_data import validate_data

SEED = 42
QUESTION = "What is the leader of Workshop W0001?"
EXTRA_PATH = Path("data/training/evidence_span_v5b_extra_train.jsonl")
FAMILIAR_5_PATH = Path("data/evaluation/evidence_span_v5b_familiar5.jsonl")
NOVEL_4_PATH = Path("data/evaluation/evidence_span_v5b_novel4.jsonl")
PATHS = {
    "extra_train": EXTRA_PATH,
    "familiar5": FAMILIAR_5_PATH,
    "novel4": NOVEL_4_PATH,
}
EXPECTED_COUNTS = {"extra_train": 72, "familiar5": 24, "novel4": 24}


def _make_rows(group: str, names: list[str], contexts_per_name: int, rng) -> list[dict]:
    rows = []
    used_information = set()
    for name in names:
        for context in rng.sample(CONTEXTS, contexts_per_name):
            information = information_for(name, context)
            if information in used_information:
                raise ValueError("Duplicate context/name evidence.")
            used_information.add(information)
            rows.append({
                "source_id": "", "example_id": "", "information": information,
                "question": QUESTION, "answerability": "answerable",
                "answer": f"Leader: {name}.",
            })
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["source_id"] = f"evidence_span_v5b_{group}_{index:04d}"
        row["example_id"] = row["source_id"] + "_q01"
    return rows


def build_additions(seed: int = SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    # Train end fragments contain only familiar V3 training syllables.
    # New lengths deliberately vary the number of bytes in the name.
    length3 = sorted({start[0] + end for start, end in product(TRAIN_START, TRAIN_END)})
    length5 = sorted({start + end + "n" for start, end in product(TRAIN_START, TRAIN_END)})
    length6 = sorted({start + end + "ra" for start, end in product(TRAIN_START, TRAIN_END)})
    rng.shuffle(length3)
    rng.shuffle(length5)
    rng.shuffle(length6)
    # Reserve complete 5-letter names before any training example is generated.
    training_names = length3[:8] + length5[:8] + length6[:8]
    dev_familiar5 = length5[8:20]
    # Unfamiliar starting syllables are never used in extra training names.
    novel4 = [start + end for start, end in product(
        UNSEEN_START, ("ze", "vo", "pe", "mi", "yn")
    )]
    rng.shuffle(novel4)
    dev_novel4 = novel4[:12]
    if (
        len(set(training_names)) != 24
        or len(set(dev_familiar5)) != 12 or len(set(dev_novel4)) != 12
        or set(training_names) & (set(dev_familiar5) | set(dev_novel4))
        or set(dev_familiar5) & set(dev_novel4)
    ):
        raise ValueError("V5B name partition collision.")
    datasets = {
        "extra_train": _make_rows("extra_train", training_names, 3, rng),
        "familiar5": _make_rows("familiar5", dev_familiar5, 2, rng),
        "novel4": _make_rows("novel4", dev_novel4, 2, rng),
    }
    return datasets


def validate_additions(group: str, rows: list[dict]) -> None:
    """Strict independent-file validation against a fixed deterministic dataset."""
    if group not in PATHS:
        raise ValueError("Unsupported V5B partition.")
    if len(rows) != EXPECTED_COUNTS[group]:
        raise ValueError(f"Incorrect V5B {group} row count.")
    expected = build_additions()[group]
    if rows != expected:
        raise ValueError(f"{group} differs from the locked seed-42 V5B dataset.")
    lengths = {}
    for row in rows:
        fields = parse_fields(row["information"], row["source_id"])
        leader = fields.get("Leader")
        if not leader or row["answer"] != f"Leader: {leader}.":
            raise ValueError("V5B evidence and answer disagree.")
        lengths[len(leader)] = lengths.get(len(leader), 0) + 1
    required = {
        "extra_train": {3: 24, 5: 24, 6: 24},
        "familiar5": {5: 24},
        "novel4": {4: 24},
    }
    if lengths != required[group]:
        raise ValueError(f"Unexpected V5B name-length distribution: {lengths}")


def read_additions(group: str) -> list[dict]:
    if group not in PATHS:
        raise ValueError(f"Unknown V5B group: {group}")
    rows, errors = validate_data(PATHS[group])
    if errors:
        raise ValueError(f"V5B {group} structural validation: " + "; ".join(errors))
    validate_additions(group, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("V5B is locked to seed 42 to prevent accidental split changes.")
    additions = build_additions()
    # Verify all outputs first. Fail before writing ANY if one file conflicts.
    pending = []
    for group, path in PATHS.items():
        rows = additions[group]
        validate_additions(group, rows)
        encoded = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        if path.exists():
            if path.read_bytes() != encoded:
                raise FileExistsError(f"Refusing to overwrite existing V5B file: {path}")
        else:
            pending.append((path, encoded))
    for path, encoded in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(encoded)
    for group, rows in additions.items():
        print(f"{group}: {len(rows)} rows => {PATHS[group]}")
    print("V3 data unchanged; V5B train addition uses 3/5/6-character familiar-pattern names.")
    print("V5B development controls: familiar 5 and novel-pattern 4; final holdout untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
