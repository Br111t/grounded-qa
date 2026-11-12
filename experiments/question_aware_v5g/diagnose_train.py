"""Read-only V5G training-set predictions, compared with an existing DEV report.

Reuses the existing decoder and scorecard; never opens the reserved EVAL
partition or updates any model, dataset, checkpoint or existing report.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.evaluate import enriched, predict, summarize
from grounded_qa.models.span_model_v5 import load_span_checkpoint


def counts_by_structure(predictions):
    result = defaultdict(lambda: {"count": 0, "correct": 0, "false_abstentions": 0})
    for row in predictions:
        fields = tuple(
            field for field in ("Leader", "Location", "Topic")
            if field + ":" in row["information"]
        )
        key = "+".join(fields)
        group = result[key]
        group["count"] += 1
        group["correct"] += int(row["correct"])
        group["false_abstentions"] += int(row["false_abstention"])
    return dict(sorted(result.items()))


def markdown_report(train_metrics, dev_metrics, train_predictions):
    def fraction(n, d):
        return f"{n}/{d} ({n/d:.1%})" if d else "n/a"

    train = train_metrics["canonical"]
    dev = dev_metrics["canonical"] if dev_metrics else None
    lines = [
        "# V5G training-set prediction diagnostic",
        "",
        "Same frozen checkpoint, same decoder, same 0.5 answerability threshold.",
        "TRAIN is in-sample. DEV is out-of-sample; no reserved EVAL was accessed.",
        "",
        "## Overall results",
        "",
        "| Metric | TRAIN | DEV |",
        "| --- | --- | --- |",
        f"| Exact accuracy | {fraction(train['correct'], train['count'])} | "
        + (fraction(dev["correct"], dev["count"]) if dev else "not supplied") + " |",
        f"| Both questions correct | {train_metrics['paired']['both_correct']}/"
        f"{train_metrics['paired']['count']} | "
        + (f"{dev_metrics['paired']['both_correct']}/{dev_metrics['paired']['count']}"
           if dev_metrics else "not supplied") + " |",
        f"| Correct switches on complete pairs | "
        f"{train_metrics['paired']['correct_switches']}/"
        f"{train_metrics['paired']['both_fields_present']} | "
        + (f"{dev_metrics['paired']['correct_switches']}/"
           f"{dev_metrics['paired']['both_fields_present']}"
           if dev_metrics else "not supplied") + " |",
        f"| False abstentions | {train['false_abstentions']} | "
        + (str(dev["false_abstentions"]) if dev else "not supplied") + " |",
        "",
        "## Training-set predictions by requested field",
        "",
        "| Question | Correct | Total |",
        "| --- | ---: | ---: |",
    ]
    for field in ("leader", "location"):
        group = train_metrics[f"{field}_canonical"]
        lines.append(f"| {field.title()} | {group['correct']} | {group['total']} |")
    lines += [
        "",
        "## Training-set confusion counts",
        "",
        "| Requested | Predicted | Count |",
        "| --- | --- | ---: |",
    ]
    confusion = Counter(
        (r["target_field"], r["predicted_field"] or "abstain")
        for r in train_predictions
    )
    for (requested, selected), n in sorted(confusion.items()):
        lines.append(f"| {requested} | {selected} | {n} |")
    lines += ["", "## Training-set performance by evidence structure", "",
              "| Fields available | Correct | Questions | False abstentions |",
              "| --- | ---: | ---: | ---: |"]
    for structure, data in counts_by_structure(train_predictions).items():
        lines.append(
            f"| {structure} | {data['correct']} | {data['count']} | "
            f"{data['false_abstentions']} |"
        )
    lines += [
        "",
        "## Paired question comparison (same evidence)",
        "",
        "| Source | Fields present | Leader requested → selected | "
        "Location requested → selected | Both correct |",
        "| --- | --- | --- | --- | --- |",
    ]
    paired = defaultdict(dict)
    for row in train_predictions:
        paired[row["source_id"]][row["target_field"]] = row
    for source, pair in sorted(paired.items()):
        lead, loc = pair["leader"], pair["location"]
        available = ", ".join(
            field for field in ("Leader", "Location")
            if field + ":" in lead["information"]
        )
        lines.append(
            f"| {source} | {available} | "
            f"{lead['predicted_field'] or 'abstain'} | "
            f"{loc['predicted_field'] or 'abstain'} | "
            f"{'yes' if lead['correct'] and loc['correct'] else 'no'} |"
        )
    lines += [
        "",
        "## How to interpret",
        "",
        "- TRAIN strong, DEV weak: suspect generalization, value patterns, or evidence-format shortcuts.",
        "- TRAIN weak, DEV weak: first inspect training targets, optimization, sampling, and decoder behavior.",
        "- Both-correct accuracy matters more than overall accuracy for question-aware field selection.",
        "- Candidate logits and answerability scores for every row are in predictions.json.",
        "- Training-set performance is diagnostic, not an estimate of generalization.",
        "- This script does not alter any existing checkpoint or consume the reserved evaluation set.",
        "",
    ]
    return "\n".join(lines)


def run(checkpoint, output_dir, dev_metrics_path=None):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_dir}")
    rows = read_partition("train")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    train_sha = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    model, metadata = load_span_checkpoint(checkpoint)
    if (metadata.get("experiment") != "v5g_leader_location_question_selection"
            or metadata.get("training_sha256") != train_sha):
        raise ValueError("Checkpoint does not match registered V5G training data")
    dev_metrics = None
    if dev_metrics_path is not None:
        dev_report = json.loads(dev_metrics_path.read_text(encoding="utf-8"))
        if (dev_report.get("phase") != "dev"
                or dev_report.get("checkpoint_sha256") != checkpoint_sha):
            raise ValueError("DEV report does not match this checkpoint")
        dev_metrics = dev_report["metrics"]

    torch.set_num_threads(1)
    model.eval()
    predictions = [enriched(row, predict(model, row)) for row in rows]
    metrics = summarize(predictions)
    report = {
        "phase": "train_diagnostic",
        "checkpoint_sha256": checkpoint_sha,
        "training_data_sha256": train_sha,
        "metrics": metrics,
        "by_evidence_structure": counts_by_structure(predictions),
        "in_sample": True,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        markdown_report(metrics, dev_metrics, predictions), encoding="utf-8"
    )
    print(json.dumps({
        "train_correct": metrics["canonical"]["correct"],
        "train_total": metrics["canonical"]["count"],
        "train_both_correct": metrics["paired"]["both_correct"],
        "train_correct_switches": metrics["paired"]["correct_switches"],
        "train_false_abstentions": metrics["canonical"]["false_abstentions"],
        "summary_path": str(output_dir / "summary.md"),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-metrics", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output_dir, args.dev_metrics)


if __name__ == "__main__":
    main()
