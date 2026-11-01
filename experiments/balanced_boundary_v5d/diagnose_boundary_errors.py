"""Read-only per-example boundary diagnostics for V5, V5B and V5D.

Evaluate exactly the same locked V5D DEV records for all three checkpoints.
Gold reference offsets are derived only AFTER prediction and never sent to the
model. Offsets below are inclusive token/byte positions, not character offsets.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.balanced_boundary_v5d.generate_data_v5d import PATHS, read_split
from experiments.balanced_boundary_v5d.evaluate_v5d import grouped
from grounded_qa.models.span_model_v5 import (
    load_span_checkpoint, predict_span, prepare_span_input, training_span_target,
)

GROUP_ORDER = (
    "length4_vowel", "length4_consonant",
    "length5_vowel", "length5_consonant",
)
FIELDS = (
    "model", "group", "example_id", "expected", "predicted", "probability",
    "gold_start", "gold_end", "raw_start", "raw_end",
    "raw_start_delta", "raw_end_delta",
    "selected_start", "selected_end", "selected_start_delta",
    "selected_end_delta", "exact", "raw_start_correct",
    "raw_end_correct", "selected_span_correct", "abstained",
)


def summarize(records):
    """Summarize observed behavior, not presumed error mechanisms."""
    n = len(records)
    if n == 0:
        raise ValueError("Cannot summarize empty predictions")
    return {
        "total": n,
        "exact": sum(r["exact"] for r in records),
        "raw_start_correct": sum(r["raw_start_correct"] for r in records),
        "raw_end_correct": sum(r["raw_end_correct"] for r in records),
        "selected_span_correct": sum(r["selected_span_correct"] for r in records),
        "abstentions": sum(r["abstained"] for r in records),
        "raw_end_deltas": dict(sorted(Counter(
            str(r["raw_end_delta"]) for r in records
        ).items(), key=lambda item: int(item[0]))),
        "selected_end_deltas": dict(sorted(Counter(
            str(r["selected_end_delta"]) for r in records
            if r["selected_end_delta"] is not None
        ).items(), key=lambda item: int(item[0]))),
        "raw_end_correct_but_selected_wrong": sum(
            r["raw_end_correct"] and not r["selected_span_correct"] for r in records
        ),
        "raw_end_wrong_but_selected_correct": sum(
            not r["raw_end_correct"] and r["selected_span_correct"] for r in records
        ),
    }


def evaluate_one(model_name, checkpoint, rows, dev_sha):
    model, info = load_span_checkpoint(checkpoint)
    if info.get("answerability_threshold") != 0.5:
        raise ValueError(f"{model_name}: unexpected answerability threshold")
    if model_name == "v5d":
        if info.get("experiment") != "v5d_balanced_boundaries":
            raise ValueError("V5D checkpoint has incorrect experiment identity")
        train_hash = hashlib.sha256(PATHS["train"].read_bytes()).hexdigest()
        if info.get("extra_training_sha256") != train_hash:
            raise ValueError("V5D checkpoint training fingerprint mismatch")
    else:
        if info.get("experiment") == "v5d_balanced_boundaries":
            raise ValueError(f"{model_name}: cannot masquerade V5D as baseline")
    records = []
    for group, cell in grouped(rows).items():
        for row in cell:
            # No reference labels supplied to inference.
            predicted, probability, selected = predict_span(
                model, row["information"], row["question"], threshold=0.5
            )
            tokens, mask = prepare_span_input(row["information"], row["question"])
            with torch.no_grad():
                starts, ends, _ = model(tokens, mask)
            raw_start = int(starts[0].argmax().item())
            raw_end = int(ends[0].argmax().item())
            # This is scoring ONLY, after the model's prediction.
            answerable, gold_start, gold_end = training_span_target(row, tokens, mask)
            if not answerable:
                raise ValueError("V5D DEV unexpectedly contains an unanswerable case")
            start, end = selected if selected is not None else (None, None)
            records.append({
                "model": model_name, "group": group,
                "example_id": row["example_id"],
                "expected": row["answer"], "predicted": predicted,
                "probability": round(probability, 6),
                "gold_start": gold_start, "gold_end": gold_end,
                "raw_start": raw_start, "raw_end": raw_end,
                "raw_start_delta": raw_start - gold_start,
                "raw_end_delta": raw_end - gold_end,
                "selected_start": start, "selected_end": end,
                "selected_start_delta": None if start is None else start - gold_start,
                "selected_end_delta": None if end is None else end - gold_end,
                "exact": int(predicted == row["answer"]),
                "raw_start_correct": int(raw_start == gold_start),
                "raw_end_correct": int(raw_end == gold_end),
                "selected_span_correct": int(selected == (gold_start, gold_end)),
                "abstained": int(selected is None),
            })
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_metadata": info,
        "dev_sha256": dev_sha,
        "overall": summarize(records),
        "groups": {group: summarize([r for r in records if r["group"] == group])
                   for group in GROUP_ORDER},
    }, records


def format_summary(results):
    lines = [
        "# V5D boundary error diagnostic",
        "",
        "**Read-only · V5D development data · 96 examples per checkpoint**",
        "",
        "## Overall",
        "",
        "| Model | Exact | Raw start | Raw end | Selected span | Abstentions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, data in results.items():
        s = data["overall"]
        lines.append(
            f"| {name.upper()} | {s['exact']}/96 | {s['raw_start_correct']}/96 | "
            f"{s['raw_end_correct']}/96 | {s['selected_span_correct']}/96 | "
            f"{s['abstentions']} |"
        )
    lines += [
        "", "## Per-condition breakdown", "",
        "| Model | Group | Exact | Raw start | Raw end |",
        "|---|---|---:|---:|---:|",
    ]
    for name, data in results.items():
        for group in GROUP_ORDER:
            s = data["groups"][group]
            lines.append(
                f"| {name.upper()} | {group} | {s['exact']}/24 | "
                f"{s['raw_start_correct']}/24 | {s['raw_end_correct']}/24 |"
            )
    lines += ["", "## End-position offset distributions", "",
              "Offsets are predicted end minus gold end, measured in model byte-token "
              "positions (inclusive). Negative = early; positive = late. "
              "Selected offsets exclude abstentions.", ""]
    for name, data in results.items():
        lines.append(f"### {name.upper()}")
        for group in GROUP_ORDER:
            s = data["groups"][group]
            lines.append(
                f"- {group}: raw {s['raw_end_deltas']}; "
                f"selected {s['selected_end_deltas']}"
            )
        s = data["overall"]
        lines.append(
            f"- Raw end correct but selected span wrong: "
            f"{s['raw_end_correct_but_selected_wrong']}; "
            f"raw end wrong but selected span correct: "
            f"{s['raw_end_wrong_but_selected_correct']}"
        )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "**Observation:** Compare offset concentration and disagreements between "
        "raw heads and jointly decoded spans. See `predictions.csv` for individual cases.",
        "",
        "**Hypotheses to assess:** A consistent -1 offset could indicate a "
        "systematic early boundary; widely dispersed offsets would suggest "
        "a broader position-selection problem. Neither is proved by these metrics alone.",
        "",
        "**Next test:** Design a new pre-registered holdout or a controlled training "
        "experiment only after reviewing these errors. Avoid repeatedly optimizing "
        "against this DEV set.",
        "",
        "No training, threshold tuning, V5C reuse or final-holdout access.",
        "",
    ]
    return "\n".join(lines)


def run(checkpoints, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {output_dir}")
    rows = read_split("dev")
    dev_sha = hashlib.sha256(PATHS["dev"].read_bytes()).hexdigest()
    results, predictions = {}, []
    for name, path in checkpoints.items():
        result, records = evaluate_one(name, path, rows, dev_sha)
        results[name] = result
        predictions.extend(records)
    payload = {
        "experiment": "v5d_boundary_error_diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "v5d_dev_only",
        "dev_sha256": dev_sha,
        "models": results,
    }
    markdown = format_summary(results)
    # All inference finishes before any artifact is written.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(predictions)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved diagnostic to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v5", "v5b", "v5d"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        run({name: getattr(args, name) for name in ("v5", "v5b", "v5d")},
            args.output_dir)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Boundary diagnostic stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
