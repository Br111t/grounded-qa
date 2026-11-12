"""V5G registered dataset design; no prior evaluation examples are imported.

Train: F-initial names and canonical Leader/Location questions only.
DEV: G-initial names, canonical questions only.
SEALED eval: H-initial names and previously unused paraphrases, unrelated
questions and Topic transfer; NOT loaded by training code.
"""
import hashlib
import json
import random
from pathlib import Path

from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import ROOMS, TOPICS

ROOT = Path("experiments/question_aware_v5g")
DATA = ROOT / "data"
QUESTIONS = {
    "leader": "What is the leader of Workshop W0001?",
    "location": "Where is Workshop W0001 located?",
}
EVAL_QUESTIONS = {
    "leader": ("Who is responsible for leading Workshop W0001?",
               "Name the person directing Workshop W0001."),
    "location": ("In which room does Workshop W0001 take place?",
                 "Identify the room assigned to Workshop W0001."),
    "topic": ("Which subject is taught at Workshop W0001?",),
    "unrelated": ("What is the price of a train ticket tomorrow?",),
}
SEEDS = {"train": 511, "dev": 512, "eval": 513}
INITIALS = {"train": "F", "dev": "G", "eval": "H"}


def build_evidence(leader, location, topic, reverse):
    fields = []
    if leader is not None:
        fields.append(("Leader", leader))
    if location is not None:
        fields.append(("Location", location))
    if topic is not None:
        fields.append(("Topic", topic))
    if reverse:
        fields.reverse()
    return "Fictional Workshop W0001. " + " ".join(
        f"{label}: {value}." for label, value in fields
    )


def make_row(partition, evidence_id, info, target, question, variant):
    import re
    fields = dict(re.findall(r"(Leader|Location|Topic): ([^.]+)\.", info))
    value = fields.get(target.title())
    return {
        "source_id": evidence_id,
        "example_id": f"{evidence_id}_{variant}_{target}",
        "information": info,
        "question": question,
        "target_field": target,
        "question_variant": variant,
        "answerability": "answerable" if value is not None else "unanswerable",
        "answer": f"{target.title()}: {value}." if value is not None
                  else "Insufficient information.",
        "expected_value": value,
    }


def build_partition(partition):
    if partition not in INITIALS:
        raise ValueError("Unregistered partition")
    rng = random.Random(SEEDS[partition])
    rows = []
    # 24 evidence sources per partition; all source identities are disjoint.
    # Each source gets a canonical Leader/Location pair.
    for i in range(24):
        leader = INITIALS[partition] + "b" + chr(97 + i) + ("a" if i % 2 else "t")
        # Assign a unique realistic room identifier to each source across splits.
        # Missing-Leader examples otherwise repeat verbatim across partitions.
        room_number = i + 1 + 24 * ("train", "dev", "eval").index(partition)
        location = f"{ROOMS[i % len(ROOMS)]} {room_number}"
        topic = ("Python", "Astronomy", "Gardening")[i % 3]
        # Balanced missing target: 6 missing Leader, 6 missing Location, 12 both.
        leader = None if i % 4 == 0 else leader
        location = None if i % 4 == 1 else location
        evidence_id = f"v5g_{partition}_{i:02d}"
        info = build_evidence(leader, location, topic, reverse=bool(i % 2))
        if partition in ("train", "dev"):
            for field in QUESTIONS:
                rows.append(make_row(partition, evidence_id, info, field,
                                     QUESTIONS[field], "canonical"))
        else:
            # New evaluation evidence; each test category is reported separately.
            for field in QUESTIONS:
                rows.append(make_row(partition, evidence_id, info, field,
                                     QUESTIONS[field], "canonical"))
                for j, question in enumerate(EVAL_QUESTIONS[field]):
                    rows.append(make_row(partition, evidence_id, info, field,
                                         question, f"paraphrase{j+1}"))
            for question in EVAL_QUESTIONS["topic"]:
                rows.append(make_row(partition, evidence_id, info, "topic",
                                     question, "transfer"))
            for question in EVAL_QUESTIONS["unrelated"]:
                rows.append(make_row(partition, evidence_id, info, "unrelated",
                                     question, "unsupported"))
    rng.shuffle(rows)
    if len(rows) != {"train": 48, "dev": 48, "eval": 192}[partition]:
        raise ValueError("Unexpected dataset row count")
    if len({r["source_id"] for r in rows}) != 24:
        raise ValueError("Expected 24 independent evidence identities")
    if len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate example identifiers")
    if partition != "eval" and any(r["target_field"] not in QUESTIONS
                                    or r["question_variant"] != "canonical"
                                    for r in rows):
        raise ValueError("Noncanonical question leaked into training/development")
    if partition == "train" and any("Topic:" not in r["information"] for r in rows):
        raise ValueError("Topic should occur as a distractor, never as a train target")
    return rows


def canonical_bytes(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")


def write_partition(partition):
    path = DATA / f"{partition}.jsonl"
    content = canonical_bytes(build_partition(partition))
    DATA.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("Refusing to overwrite changed dataset")
    else:
        with path.open("xb") as f:
            f.write(content)
    return path, hashlib.sha256(content).hexdigest()


def read_partition(partition):
    path = DATA / f"{partition}.jsonl"
    expected = canonical_bytes(build_partition(partition))
    if path.read_bytes() != expected:
        raise ValueError(f"Dataset mismatch: {partition}")
    return [json.loads(line) for line in expected.decode("utf-8").splitlines()]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", choices=("train", "dev", "eval"), required=True)
    args = ap.parse_args()
    path, digest = write_partition(args.partition)
    print(path, digest)


if __name__ == "__main__":
    main()
