"""Read-only paired end-head inspection using locked V5C development rows.

Run:
    python -m experiments.span_boundary_probe_v5c.inspect_paired_end_logits_v5c

No training, dataset generation, threshold changes, or final holdout access.
"""
from pathlib import Path

import torch

from experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c import _check_checkpoint
from experiments.span_boundary_probe_v5c.generate_span_probe_v5c import read_probe_group
from grounded_qa.models.span_model_v5 import (
    load_span_checkpoint, prepare_span_input, training_span_target,
)

CHECKPOINTS = (
    Path("experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt"),
    Path("experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt"),
)


def inspect(model, row):
    tokens, mask = prepare_span_input(row["information"], row["question"])
    answerable, start, end = training_span_target(row, tokens, mask)
    if answerable != 1.0:
        raise ValueError("Expected an answerable locked probe.")
    with torch.no_grad():
        _, logits, _ = model(tokens, mask)
        probs = logits[0].softmax(dim=-1)
    candidate = logits[0]
    best_end = int(candidate.argmax().item())
    rank = 1 + int((candidate[mask[0]] > candidate[end]).sum().item())
    name = row["answer"].removeprefix("Leader: ").removesuffix(".")
    return {
        "name": name, "start": start, "end": end, "best_end": best_end,
        "rank": rank, "gold_prob": float(probs[end].item()),
        "best_prob": float(probs[best_end].item()),
        "context": row["information"],
    }


def main():
    torch.set_num_threads(1)
    groups = {
        group: read_probe_group(group)
        for group in (
            "length4_vowel", "length4_consonant",
            "length5_vowel", "length5_consonant",
        )
    }
    if any(len(rows) != 24 for rows in groups.values()):
        raise ValueError("Expected exactly 24 locked examples per group.")
    print("READ ONLY: paired V5C end-head comparison; gold indices used for diagnostics only.")
    print("Each row compares names with the SAME context and length; only final character differs.")
    for checkpoint in CHECKPOINTS:
        model, info = load_span_checkpoint(checkpoint)
        _check_checkpoint(info)
        model.eval()
        print(f"\n=== {checkpoint} ===")
        for length in (4, 5):
            vowels = groups[f"length{length}_vowel"]
            consonants = groups[f"length{length}_consonant"]
            print(f"\nLength {length}: vowel -> consonant (24 matched rows)")
            print("row | names | gold ends | independent end | gold rank | P(gold) | P(best)")
            changed = 0
            for index, (vowel_row, consonant_row) in enumerate(zip(vowels, consonants), 1):
                a = inspect(model, vowel_row)
                b = inspect(model, consonant_row)
                if a["name"][:-1] != b["name"][:-1]:
                    raise ValueError(f"Unmatched name stem at row {index}.")
                if a["context"].replace(a["name"], "<NAME>") != b["context"].replace(b["name"], "<NAME>"):
                    raise ValueError(f"Unmatched source context at row {index}.")
                if a["start"] != b["start"] or a["end"] != b["end"]:
                    raise ValueError(f"Changed boundary positions at row {index}.")
                changed += a["best_end"] != b["best_end"]
                print(
                    f"{index:02d} | {a['name']} -> {b['name']} | {a['end']}/{b['end']} | "
                    f"{a['best_end']} -> {b['best_end']} | {a['rank']} -> {b['rank']} | "
                    f"{a['gold_prob']:.6g} -> {b['gold_prob']:.6g} | "
                    f"{a['best_prob']:.6g} -> {b['best_prob']:.6g}"
                )
            print(f"Independent end changed for {changed}/24 matched final-character swaps.")
    print("\nNo files, checkpoints, or weights changed.")


if __name__ == "__main__":
    main()
