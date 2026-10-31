"""Read-only V5C end-boundary error diagnostic for V5 and V5B checkpoints.

Run from the repository root:
    $env:PYTHONPATH = "$PWD\src"
    python -m experiments.span_boundary_probe_v5c.diagnose_end_boundary_errors_v5c

Uses the existing, locked V5C development probes only. Does not train,
change model weights, regenerate datasets, or access final-holdout files.
"""

from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.span_boundary_probe_v5c.generate_span_probe_v5c import GROUPS, read_probe_group
from experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c import _check_checkpoint
from grounded_qa.models.span_model_v5 import (
    load_span_checkpoint, prepare_span_input, predict_span, training_span_target,
)

CHECKPOINTS = (
    Path("experiments/span_extractor_v5/checkpoints/evidence_span_v5_3000_steps.pt"),
    Path("experiments/variable_span_v5b/checkpoints/evidence_span_v5b_3000_steps.pt"),
)


def inspect_checkpoint(path: Path, groups: dict[str, list[dict]]) -> None:
    model, info = load_span_checkpoint(path)
    _check_checkpoint(info)
    model.eval()

    print(f"\n=== {path} ({info.get('experiment', 'v5_original')}) ===")
    for group in GROUPS:
        records = []
        for row in groups[group]:
            tokens, mask = prepare_span_input(row["information"], row["question"])
            answerable, gold_start, gold_end = training_span_target(row, tokens, mask)
            if answerable != 1.0:
                raise ValueError("V5C probes must all be answerable")
            with torch.no_grad():
                start_logits, end_logits, _ = model(tokens, mask)
                end_probs = end_logits[0].softmax(dim=-1)
            output, confidence, selected = predict_span(
                model, row["information"], row["question"], threshold=0.5
            )
            independent_start = int(start_logits[0].argmax().item())
            independent_end = int(end_logits[0].argmax().item())
            top5 = torch.topk(end_probs, k=min(5, int(mask[0].sum().item()))).indices.tolist()
            name = row["answer"][len("Leader: "):-1]
            records.append({
                "id": row["example_id"], "name": name,
                "gold_start": gold_start, "gold_end": gold_end,
                "independent_start": independent_start, "independent_end": independent_end,
                "selected": selected, "prediction": output,
                "top5": top5,
                "gold_rank": int((end_logits[0] > end_logits[0, gold_end]).sum().item()) + 1,
                "gold_prob": float(end_probs[gold_end].item()),
            })
        count = len(records)
        exact = sum(x["prediction"] == f"Leader: {x['name']}." for x in records)
        head_correct = sum(x["independent_end"] == x["gold_end"] for x in records)
        selected_correct = sum(x["selected"] is not None and x["selected"][1] == x["gold_end"] for x in records)
        top5_correct = sum(x["gold_end"] in x["top5"] for x in records)
        changed = sum(x["selected"] is not None and x["selected"][1] != x["independent_end"] for x in records)
        abstained = sum(x["selected"] is None for x in records)
        offsets = Counter(x["independent_end"] - x["gold_start"] for x in records)
        errors = Counter(x["independent_end"] - x["gold_end"] for x in records)
        selected_lengths = Counter((x["selected"][1] - x["selected"][0] + 1) if x["selected"] is not None else "abstain" for x in records)
        by_char = defaultdict(list)
        for x in records:
            by_char[x["name"][-1]].append(x)
        print(f"\n{group}: n={count}; exact={exact}/{count}; independent end={head_correct}/{count}; "
              f"selected end={selected_correct}/{count}; gold end in top5={top5_correct}/{count}")
        print(f"  independent end offset from GOLD START: {dict(sorted(offsets.items()))}")
        print(f"  independent end error (predicted - GOLD END): {dict(sorted(errors.items()))}")
        print(f"  final selected span lengths: {dict(sorted(selected_lengths.items(), key=lambda i: str(i[0])))}")
        print(f"  joint selection changed independent end: {changed}/{count}; abstained: {abstained}/{count}")
        for char, matches in sorted(by_char.items()):
            print(f"  ending {char!r}: independent end correct "
                  f"{sum(x['independent_end'] == x['gold_end'] for x in matches)}/{len(matches)}, "
                  f"gold end top5 {sum(x['gold_end'] in x['top5'] for x in matches)}/{len(matches)}")
        examples = [x for x in records if x["independent_end"] != x["gold_end"]][:2]
        for x in examples:
            print(f"  example {x['id']}: name={x['name']!r} gold=({x['gold_start']},{x['gold_end']}) "
                  f"head=({x['independent_start']},{x['independent_end']}) "
                  f"selected={x['selected']} gold_rank={x['gold_rank']} "
                  f"gold_probability={x['gold_prob']:.8f} top5={x['top5']}")


def main() -> None:
    torch.set_num_threads(1)
    groups = {group: read_probe_group(group) for group in GROUPS}
    assert all(len(rows) == 24 for rows in groups.values()), "Unexpected V5C probe sizes"
    print("READ ONLY: V5C development diagnostic; no training, writes, or final holdout access.")
    for path in CHECKPOINTS:
        inspect_checkpoint(path, groups)


if __name__ == "__main__":
    main()
