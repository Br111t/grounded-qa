"""V5D controlled balanced boundary data. Never reuse V5C development rows.

Two distinct name-stem families (C.. training, D.. development) produce
96 answerable rows per partition: 24 for each (length x ending) cell.
The base V3 168 training rows, including missing-evidence cases, are retained.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import (
    CONTEXTS, QUESTION, information_for,
)
from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields
from experiments.span_boundary_probe_v5c.generate_span_probe_v5c import build_probe_datasets
from experiments.variable_span_v5b.generate_span_data_v5b import build_additions
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import build_datasets
from scripts.data.validate_data import validate_data

SEED = 42
GROUPS = ("length4_vowel", "length4_consonant", "length5_vowel", "length5_consonant")
ROOT = Path("experiments/balanced_boundary_v5d/data")
PATHS = {"train": ROOT / "v5d_balanced_train.jsonl",
         "dev": ROOT / "v5d_balanced_dev.jsonl"}
MANIFEST = ROOT / "manifest.json"


def build_split(partition: str) -> list[dict]:
    if partition not in PATHS:
        raise ValueError("Unregistered V5D split")
    rng = random.Random(SEED + (0 if partition == "train" else 1))
    # Names deliberately use different initial-character families. The
    # first character makes them disjoint from existing V3/V5B/V5C names.
    initial = "C" if partition == "train" else "D"
    stems = [initial + "b" + letter for letter in "abcdefghijkl"]
    rows = []
    for stem_index, stem in enumerate(stems):
        vowel, consonant = (("a", "n") if stem_index % 2 == 0 else ("i", "t"))
        names = {
            "length4_vowel": stem + vowel,
            "length4_consonant": stem + consonant,
            "length5_vowel": stem + "e" + vowel,
            "length5_consonant": stem + "e" + consonant,
        }
        contexts = rng.sample(CONTEXTS, 2)
        for context_index, context in enumerate(contexts):
            for group in GROUPS:
                name = names[group]
                identifier = f"v5d_{partition}_{group}_{stem_index:02d}_{context_index}"
                rows.append({
                    "source_id": identifier,
                    "example_id": identifier + "_q01",
                    "information": information_for(name, context),
                    "question": QUESTION,
                    "answerability": "answerable",
                    "answer": f"Leader: {name}.",
                })
    rng.shuffle(rows)
    validate_split(partition, rows)
    return rows


def validate_split(partition: str, rows: list[dict]) -> None:
    if partition not in PATHS:
        raise ValueError("Unregistered V5D split")
    counts = dict.fromkeys(GROUPS, 0)
    names = set()
    for row in rows:
        fields = parse_fields(row["information"], row["source_id"])
        name = fields.get("Leader")
        if not name or row["answerability"] != "answerable" or row["answer"] != f"Leader: {name}.":
            raise ValueError("Invalid V5D training reference")
        if not name.startswith("C" if partition == "train" else "D"):
            raise ValueError("V5D partition stem leaked")
        group = f"length{len(name)}_{'vowel' if name[-1].lower() in 'aeiou' else 'consonant'}"
        if group not in counts:
            raise ValueError("Unexpected name length or ending")
        counts[group] += 1
        names.add(name)
    if counts != dict.fromkeys(GROUPS, 24) or len(names) != 48 or len(rows) != 96:
        raise ValueError(f"V5D unbalanced or duplicate names: {counts}")
    if len({row["information"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evidence inside V5D split")


def validate_disjoint(train: list[dict], dev: list[dict]) -> None:
    def names(rows):
        return {parse_fields(r["information"], r["source_id"]).get("Leader") for r in rows}
    existing = build_datasets()
    additions = build_additions()
    probe = build_probe_datasets()
    prior_names = set().union(*(names(v) for v in
        [*existing.values(), *additions.values(), *probe.values()]))
    if names(train) & (names(dev) | prior_names) or names(dev) & prior_names:
        raise ValueError("V5D names overlap an existing data partition")
    if ({r["information"] for r in train} & {r["information"] for r in dev}):
        raise ValueError("V5D splits share evidence")


def encoded(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")


def read_split(partition: str) -> list[dict]:
    if partition not in PATHS:
        raise ValueError("Unregistered V5D split")
    rows, errors = validate_data(PATHS[partition])
    if errors:
        raise ValueError("Invalid V5D file: " + "; ".join(errors))
    expected = build_split(partition)
    if rows != expected:
        raise ValueError("V5D data differ from fixed seed-42 design")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        parser.error("V5D is locked to seed 42")
    train, dev = build_split("train"), build_split("dev")
    validate_disjoint(train, dev)
    payloads = {PATHS["train"]: encoded(train), PATHS["dev"]: encoded(dev)}
    manifest = {
        "experiment": "v5d_balanced_boundaries",
        "seed": SEED,
        "design": "2 lengths x 2 ending classes, 24 rows per cell in each partition",
        "train_sha256": hashlib.sha256(payloads[PATHS["train"]]).hexdigest(),
        "dev_sha256": hashlib.sha256(payloads[PATHS["dev"]]).hexdigest(),
        "v5c_reuse": False,
    }
    payloads[MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    # Preflight all destinations to avoid replacing or partially overwriting data.
    for path, data in payloads.items():
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    ROOT.mkdir(parents=True, exist_ok=True)
    for path, data in payloads.items():
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    print("V5D generated: 96 balanced TRAIN + 96 matched DEV; V5C untouched")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
