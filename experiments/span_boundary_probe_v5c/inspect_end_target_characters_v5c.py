"""Read-only V5C diagnostic: map independent end-head predictions to evidence characters.

Run from repository root with PYTHONPATH pointing to src:
    python -m experiments.span_boundary_probe_v5c.inspect_end_target_characters_v5c

Uses only locked V5C development probes and existing V5/V5B checkpoints.
Does not train, write data, or access final holdout.
"""
from collections import Counter
from pathlib import Path

import torch

from experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c import _check_checkpoint
from experiments.span_boundary_probe_v5c.generate_span_probe_v5c import GROUPS, read_probe_group
from grounded_qa.models.span_model_v5 import (
    SOURCE_PREFIX, load_span_checkpoint, prepare_span_input, training_span_target,
)
from grounded_qa.tokenization.tokenizer import encode

CHECKPOINTS = (
    Path("experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt"),
    Path("experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt"),
)


def locate_endpoint(information: str, position: int, gold_start: int, gold_end: int):
    """Return region and source character for a model token position.

    Token position zero is BOS; remaining positions count UTF-8 bytes.
    Locked V5C evidence is ASCII, so each source byte maps to one character.
    """
    source_start = 1 + len(encode(SOURCE_PREFIX))
    source_bytes = information.encode("utf-8")
    offset = position - source_start
    if not 0 <= offset < len(source_bytes):
        raise ValueError(f"Predicted endpoint {position} falls outside evidence.")
    character = chr(source_bytes[offset])
    if gold_start <= position < gold_end:
        region = "inside_name"
    elif position == gold_end:
        region = "correct_name_end"
    else:
        # Identify the source field rather than guessing from absolute positions.
        leader_marker = information.index("Leader: ")
        leader_name_start = leader_marker + len("Leader: ")
        leader_dot = information.index(".", leader_name_start)
        if offset == leader_dot:
            region = "name_punctuation"
        else:
            region = "other_evidence"
        if region == "other_evidence":
            for field in ("Location: ", "Topic: ", "Leader: "):
                start = information.find(field)
                if start < 0:
                    continue
                field_end = information.find(".", start)
                if field_end < 0:
                    continue
                if start <= offset <= field_end:
                    region = (
                        "name_field_other" if field == "Leader: "
                        else field.split(":")[0].lower() + "_field"
                    )
                    break
            if region == "other_evidence" and character.isspace():
                region = "separator"
            if region == "other_evidence" and character in ".:,": 
                region = "other_punctuation"
    return region, character, offset


def inspect_group(model, rows):
    regions = Counter()
    characters = Counter()
    repeated_positions = Counter()
    wrong_examples = []
    for row in rows:
        tokens, mask = prepare_span_input(row["information"], row["question"])
        answerable, start, end = training_span_target(row, tokens, mask)
        if answerable != 1.0:
            raise ValueError("Locked V5C row must be answerable.")
        with torch.no_grad():
            _, end_logits, _ = model(tokens, mask)
        position = int(end_logits[0].argmax().item())
        if not mask[0, position].item():
            raise AssertionError("End-head argmax must be in candidate evidence.")
        region, character, offset = locate_endpoint(row["information"], position, start, end)
        regions[region] += 1
        if position != end:
            characters[character] += 1
            repeated_positions[position] += 1
            if len(wrong_examples) < 4:
                name = row["answer"][len("Leader: "):-1]
                source_bytes = row["information"].encode("utf-8")
                left = max(0, offset - 14)
                right = min(len(source_bytes), offset + 15)
                snippet = source_bytes[left:right].decode("utf-8")
                wrong_examples.append(
                    (name, end, position, offset, character, region, snippet)
                )
    return regions, characters, repeated_positions, wrong_examples


def main():
    torch.set_num_threads(1)
    groups = {group: read_probe_group(group) for group in GROUPS}
    if any(len(rows) != 24 for rows in groups.values()):
        raise ValueError("V5C requires exactly 24 locked rows per group.")
    print("READ ONLY: independent end-head positions in source evidence; no writes or holdout access.")
    for path in CHECKPOINTS:
        model, info = load_span_checkpoint(path)
        _check_checkpoint(info)
        model.eval()
        print(f"\n=== {path} ===")
        for group in GROUPS:
            regions, chars, positions, examples = inspect_group(model, groups[group])
            print(f"\n{group} (24 rows)")
            print(f"  selected source regions: {dict(regions.most_common())}")
            print(f"  incorrect-end characters: {dict(chars.most_common())}")
            print(f"  most common incorrect absolute token positions: {positions.most_common(6)}")
            for name, gold, predicted, source_offset, character, region, snippet in examples:
                print(
                    f"  example {name!r}: gold={gold}, predicted={predicted}, "
                    f"source_byte={source_offset}, char={character!r}, region={region}, "
                    f"source_window={snippet!r}"
                )
    print("\nNo datasets, checkpoints, weights, or final-holdout files changed.")


if __name__ == "__main__":
    main()
