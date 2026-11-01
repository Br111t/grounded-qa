"""Create an independent, immutable holdout ONLY after the DEV decoder freeze.

Never print or inspect holdout results during decoder selection. The holdout
uses an E-initial name family and 12 missing-evidence cases; it is not V5C.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import (
    CONTEXTS, QUESTION, information_for,
)

SEED = 9041
HOLDOUT_PATH = Path("experiments/field_boundary_decoder_v5e/data/v5e_holdout.jsonl")


def build_holdout():
    rng = random.Random(SEED)
    rows = []
    contexts = list(CONTEXTS)
    for i, letter in enumerate("abcdefghijkl"):
        stem = "Eb" + letter
        vowel, consonant = (("a", "n") if i % 2 == 0 else ("i", "t"))
        names = (stem + vowel, stem + consonant,
                 stem + "e" + vowel, stem + "e" + consonant)
        for context_index, context in enumerate(rng.sample(contexts, 2)):
            for group_index, name in enumerate(names):
                identifier = f"v5e_holdout_{i:02d}_{context_index}_{group_index}"
                rows.append({
                    "source_id": identifier, "example_id": identifier + "_q01",
                    "information": information_for(name, context),
                    "question": QUESTION, "answerability": "answerable",
                    "answer": f"Leader: {name}.",
                })
    # Reuse distinct evidence contexts, but no names, from the answerable cases.
    missing_contexts = rng.sample(
        [c for c in CONTEXTS if c[2] is not None or not c[1]], 12
    )
    for i, context in enumerate(missing_contexts):
        identifier = f"v5e_holdout_missing_{i:02d}"
        rows.append({
            "source_id": identifier, "example_id": identifier + "_q01",
            "information": information_for(None, context),
            "question": QUESTION, "answerability": "unanswerable",
            "answer": "Insufficient information.",
        })
    rng.shuffle(rows)
    if len(rows) != 108 or len({r["information"] for r in rows}) != 108:
        raise ValueError("Holdout generation is not unique")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=HOLDOUT_PATH)
    args = parser.parse_args()
    from experiments.field_boundary_decoder_v5e.compare_decoders import fingerprints
    manifest = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (manifest.get("phase") != "dev" or
        manifest.get("dev_sha256") !=
        "5799afad25695b33eadcbbf6e224ec2467ebe6c883157cc6955a96fbdc380ecd"):
        raise ValueError("Freeze manifest missing or invalid")
    # Validate code hasn't drifted without touching checkpoint paths.
    identity = manifest["identity"]
    for file, digest in identity["code_sha256"].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
            raise ValueError("Decoder code changed after freeze")
    if args.output.exists():
        raise FileExistsError("Holdout file already exists. Never overwrite it.")
    payload = "".join(json.dumps(r, sort_keys=True) + "\n"
                      for r in build_holdout()).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print("Sealed 108 holdout records (96 answerable, 12 missing).")
    print("SHA256:", hashlib.sha256(payload).hexdigest())
    print("Do not use these examples to tune decoder rules.")


if __name__ == "__main__":
    main()
