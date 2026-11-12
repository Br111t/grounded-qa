"""Read-only V5G loss-head diagnostic: seeded initialization vs saved checkpoint.

Neither training nor evaluation partitions are changed. This measures two
endpoints, NOT intermediate training history or causal gradient interference.
Only registered TRAIN and DEV are read; reserved EVAL remains untouched.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.train import target_for
from grounded_qa.models.span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, prepare_span_input,
)


def prepared_rows(partition, span_budget):
    examples = []
    for row in read_partition(partition):
        ids, mask = prepare_span_input(row["information"], row["question"])
        target = target_for(row, ids, mask, span_budget)
        examples.append((row, ids, mask, target))
    return examples


@torch.no_grad()
def score(model, prepared):
    model.eval()  # disables dropout so comparisons are deterministic
    rows = []
    for row, ids, mask, (answerable, gold_start, gold_end) in prepared:
        starts, ends, answer_logits = model(ids, mask)
        label = torch.tensor([answerable], device=answer_logits.device)
        answer_loss = float(F.binary_cross_entropy_with_logits(answer_logits, label).item())
        probability = float(torch.sigmoid(answer_logits[0]).item())
        result = {
            "source_id": row["source_id"],
            "example_id": row["example_id"],
            "question_field": row["target_field"],
            "answerable": bool(answerable),
            "available_fields": "+".join(
                field for field in ("Leader", "Location", "Topic")
                if field + ":" in row["information"]
            ),
            "answerability_bce": answer_loss,
            "answer_probability": probability,
            "answerability_correct": (probability >= 0.5) == bool(answerable),
            "start_ce": None, "end_ce": None, "extraction_ce": None,
            "start_correct": None, "end_correct": None,
            "both_heads_correct": None,
        }
        if answerable:
            target_start = torch.tensor([gold_start], device=starts.device)
            target_end = torch.tensor([gold_end], device=ends.device)
            start_ce = float(F.cross_entropy(starts, target_start).item())
            end_ce = float(F.cross_entropy(ends, target_end).item())
            start_correct = int(starts[0].argmax().item()) == gold_start
            end_correct = int(ends[0].argmax().item()) == gold_end
            result.update({
                "start_ce": start_ce,
                "end_ce": end_ce,
                "extraction_ce": (start_ce + end_ce) / 2,
                "start_correct": start_correct,
                "end_correct": end_correct,
                "both_heads_correct": start_correct and end_correct,
            })
        rows.append(result)
    return rows


def mean(rows, field):
    valid = [r[field] for r in rows if r[field] is not None]
    return sum(valid) / len(valid) if valid else None


def aggregate(rows):
    answerable = [r for r in rows if r["answerable"]]
    return {
        "rows": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(rows) - len(answerable),
        "answerability_bce_all": mean(rows, "answerability_bce"),
        "answerability_accuracy_all": mean(rows, "answerability_correct"),
        "answerability_bce_answerable": mean(answerable, "answerability_bce"),
        "answerability_bce_unanswerable": mean(
            [r for r in rows if not r["answerable"]], "answerability_bce"
        ),
        "start_ce_answerable": mean(answerable, "start_ce"),
        "end_ce_answerable": mean(answerable, "end_ce"),
        "extraction_ce_answerable": mean(answerable, "extraction_ce"),
        "start_head_accuracy_answerable": mean(answerable, "start_correct"),
        "end_head_accuracy_answerable": mean(answerable, "end_correct"),
        "both_heads_accuracy_answerable": mean(answerable, "both_heads_correct"),
        # Matches the existing loss definition: extraction contributes zero
        # on unanswerable examples; report this separately from conditional CE.
        "mean_total_objective_all": mean(rows, "answerability_bce") +
            sum(r["extraction_ce"] or 0.0 for r in rows) / len(rows),
    }


def group_summary(rows):
    categories = defaultdict(list)
    for row in rows:
        for group, key in (
            ("field", row["question_field"]),
            ("answerability", "answerable" if row["answerable"] else "unanswerable"),
            ("evidence_structure", row["available_fields"]),
        ):
            categories[(group, key)].append(row)
    return {
        group + ":" + key: aggregate(values)
        for (group, key), values in sorted(categories.items())
    }


def render(data):
    names = ("answerability_bce_all", "answerability_bce_answerable",
             "answerability_bce_unanswerable", "start_ce_answerable",
             "end_ce_answerable", "extraction_ce_answerable",
             "mean_total_objective_all", "answerability_accuracy_all",
             "start_head_accuracy_answerable", "end_head_accuracy_answerable",
             "both_heads_accuracy_answerable")
    def fmt(value):
        return "n/a" if value is None else f"{value:.5f}"
    lines = [
        "# V5G: Answerability versus start/end losses",
        "",
        "Read-only, dropout disabled. Same registered TRAIN/DEV data,",
        "seeded pre-training initialization and saved 1,200-step checkpoint.",
        "This compares endpoints, not intermediate training steps.",
        "",
    ]
    for partition in ("train", "dev"):
        initial = data["results"][partition]["initial"]
        final = data["results"][partition]["trained"]
        lines += [
            f"## {partition.upper()} ({initial['rows']} questions)",
            "",
            "| Metric | Initialization | Trained | Change (trained − initial) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name in names:
            before, after = initial[name], final[name]
            delta = after - before if before is not None and after is not None else None
            lines.append(
                f"| {name} | {fmt(before)} | {fmt(after)} | {fmt(delta)} |"
            )
        lines += ["", "### Trained loss by question and answerability", "",
                  "| Group | Rows | Answerability BCE | Start CE | End CE |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        groups = data["groups"][partition]["trained"]
        for key, item in groups.items():
            if key.startswith("field:") or key.startswith("answerability:"):
                lines.append(
                    f"| {key} | {item['rows']} | {fmt(item['answerability_bce_all'])}"
                    f" | {fmt(item['start_ce_answerable'])} | {fmt(item['end_ce_answerable'])} |"
                )
        lines.append("")
    lines += [
        "## Interpretation guardrails",
        "",
        "- Lower cross-entropy (CE/BCE) is better; higher head accuracy is better.",
        "- Start and end CE are defined **only on answerable examples**.",
        "- The actual training objective averages start/end CE, and adds it to answerability BCE.",
        "- Mean total objective weights extraction by the fraction of answerable examples.",
        "- Start/end argmax accuracy is not equivalent to complete-field decoder accuracy.",
        "- The untrained comparison is reconstructed with the stored initial seed;",
        " it does not recover the original optimizer or intermediate checkpoints.",
        "- A decreasing loss does not establish semantic question conditioning.",
        "- Endpoint losses alone cannot prove one task's gradient updates hurt the other;",
        " gradient/ablation experiments would be needed for that causal claim.",
        "- Reserved EVAL was neither generated nor examined.",
        "",
    ]
    return "\n".join(lines)


def run(checkpoint, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    model, metadata = load_span_checkpoint(checkpoint)
    if metadata.get("experiment") != "v5g_leader_location_question_selection":
        raise ValueError("Expected V5G checkpoint")
    training_digest = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    if metadata.get("training_sha256") != training_digest:
        raise ValueError("Training dataset hash disagrees with checkpoint")
    seed = metadata["seed"]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    initialized = SpanGroundedQA(model.config).to(device="cpu", dtype=torch.float32)
    data = {"analysis": "v5g_loss_components", "seed": seed,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_sha256": training_digest, "results": {}, "groups": {}}
    for partition in ("train", "dev"):
        prepared = prepared_rows(partition, model.config.max_span_bytes)
        before = score(initialized, prepared)
        after = score(model, prepared)
        data["results"][partition] = {
            "initial": aggregate(before), "trained": aggregate(after)
        }
        data["groups"][partition] = {
            "initial": group_summary(before), "trained": group_summary(after)
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    (output / "summary.md").write_text(render(data), encoding="utf-8")
    print(render(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
