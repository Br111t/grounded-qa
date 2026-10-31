"""Evaluate fixed-question V3 evidence generation by distinct evaluation category."""

import argparse
from pathlib import Path

import torch

from grounded_qa.training.checkpoint import load_checkpoint
from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields, validate_v3_rows
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import OUTPUTS
from grounded_qa.inference.predict import generate_answer, matches_reference
from scripts.data.validate_data import validate_data

GROUPS = ("train", "known", "recombined", "unseen", "missing")


def read_partitions() -> dict[str, list[dict]]:
    groups = {}
    for group, path in OUTPUTS.items():
        rows, errors = validate_data(path)
        if errors:
            raise ValueError(f"{group} structural validation failed: {'; '.join(errors)}")
        validate_v3_rows(rows, group=group)
        groups[group] = rows

    # Check leakage before evaluation, not just within each partition.
    sources = set()
    informations = set()
    names = {}
    for group, rows in groups.items():
        names[group] = set()
        for row in rows:
            if row["source_id"] in sources or row["information"] in informations:
                raise ValueError("Repeated source or evidence across V3 partitions.")
            sources.add(row["source_id"])
            informations.add(row["information"])
            leader = parse_fields(row["information"], row["source_id"]).get("Leader")
            if leader:
                names[group].add(leader)

    if names["train"] != names["known"]:
        raise ValueError("Known-name evaluation must use all and only training names.")
    if names["train"] & (names["recombined"] | names["unseen"]):
        raise ValueError("Recombined/unseen leader values leaked into training.")
    if names["recombined"] & names["unseen"]:
        raise ValueError("Recombined and unseen leader values must be disjoint.")
    return groups


def evaluate_group(model, rows: list[dict], prompt_version: str) -> tuple[int, list[tuple]]:
    correct = 0
    failures = []
    for row in rows:
        result = generate_answer(
            model, row["information"], row["question"],
            prompt_version=prompt_version,
        )
        is_correct = matches_reference(result, row["answer"])
        correct += int(is_correct)
        if not is_correct:
            failures.append((
                row["example_id"], result.text, row["answer"], result.ended_with_eos,
            ))
    return correct, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS + ("all",), default="all")
    parser.add_argument("--show-errors", type=int, default=5)
    args = parser.parse_args()
    if args.show_errors < 0:
        parser.error("--show-errors must be nonnegative.")

    torch.set_num_threads(1)
    model, info = load_checkpoint(args.checkpoint)
    if Path(info.get("training_file", "missing")).resolve() != OUTPUTS["train"].resolve():
        raise ValueError("Checkpoint was not trained on the registered V3 training file.")

    partitions = read_partitions()
    prompt_version = info.get("prompt_version", "original")
    selected = GROUPS if args.group == "all" else (args.group,)
    print(f"Checkpoint: {args.checkpoint}; prompt: {prompt_version}")
    print("Train = fitting accuracy; other groups = development evaluations.")
    dev_correct = 0
    dev_total = 0
    for group in selected:
        rows = partitions[group]
        correct, failures = evaluate_group(model, rows, prompt_version)
        print(f"\n{group}: {correct}/{len(rows)} exact matches ({correct / len(rows):.1%})")
        for example_id, generated, expected, eos in failures[:args.show_errors]:
            print(f"  {example_id}: generated={generated!r}; expected={expected!r}; EOS={eos}")
        if len(failures) > args.show_errors:
            print(f"  ... {len(failures) - args.show_errors} further mismatches omitted")
        if group != "train":
            dev_correct += correct
            dev_total += len(rows)
    if len(selected) > 1 and dev_total:
        print(f"\nOverall development: {dev_correct}/{dev_total} ({dev_correct / dev_total:.1%})")
    print("No training performed; final holdout not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
