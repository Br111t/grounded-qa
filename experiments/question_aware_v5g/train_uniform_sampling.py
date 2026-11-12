"""Controlled V5G sampling ablation: uniform per-question shuffled epochs.

Only sampling differs from V5G baseline. Architecture, loss, decoder, data,
seed, Adam LR, and 1200 training updates remain identical. TRAIN/DEV are
measured periodically; reserved EVAL is never read. Do not overwrite files.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.diagnose_loss_components import (
    aggregate, prepared_rows, score,
)
from experiments.question_aware_v5g.evaluate import enriched, predict, summarize
from experiments.question_aware_v5g.train import target_for
from grounded_qa.models.span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, prepare_span_input,
    save_span_checkpoint, span_training_loss,
)


def uniform_schedule(n, steps, seed):
    """Every contiguous n-step epoch sees each row exactly once."""
    if n < 1 or steps < 1:
        raise ValueError("Schedule arguments must be positive")
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < steps:
        epoch = list(range(n))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:steps]


@torch.no_grad()
def snapshot(model, rows, prepared):
    model.eval()
    losses = aggregate(score(model, prepared))
    predicted = [enriched(row, predict(model, row)) for row in rows]
    metrics = summarize(predicted)
    return {
        "losses": losses,
        "exact_accuracy": metrics["canonical"]["accuracy"],
        "correct": metrics["canonical"]["correct"],
        "false_abstentions": metrics["canonical"]["false_abstentions"],
        "unsupported_answers": metrics["canonical"]["unsupported_answers"],
        "paired_both_correct": metrics["paired"]["both_correct"],
        "paired_total": metrics["paired"]["count"],
        "correct_switches": metrics["paired"]["correct_switches"],
        "both_fields_present": metrics["paired"]["both_fields_present"],
        "leader_correct": metrics["leader_canonical"]["correct"],
        "location_correct": metrics["location_canonical"]["correct"],
        "answerable_accuracy": (
            sum(r["correct"] for r in predicted if r["expected_value"] is not None)
            / sum(r["expected_value"] is not None for r in predicted)
        ),
        "unanswerable_accuracy": (
            sum(r["correct"] for r in predicted if r["expected_value"] is None)
            / sum(r["expected_value"] is None for r in predicted)
        ),
    }


def render(history, counts, training_sha):
    lines = [
        "# V5G sampling ablation: equal exposure per question",
        "",
        "From-scratch seed 42, original V5 span model, original loss, same",
        "train/dev datasets, same complete-field decoder, Adam lr=0.001.",
        "Only the example-selection schedule changes.",
        "DEV monitored for diagnosis only; reserved EVAL untouched.",
        "",
        "## Exposure counts",
        "",
        "| Category | Unique examples | Training updates |",
        "| --- | ---: | ---: |",
    ]
    for category in ("leader/answerable", "leader/unanswerable",
                     "location/answerable", "location/unanswerable"):
        num = {"leader/answerable": 18, "leader/unanswerable": 6,
               "location/answerable": 18, "location/unanswerable": 6}[category]
        lines.append(f"| {category} | {num} | {counts[category]} |")
    lines += ["", f"Training SHA256: \`{training_sha}\`", "",
              "## Whole-split diagnostics (not single-step training losses)", "",
              "| Step | Partition | Answer BCE | Start CE | End CE | Both correct pairs | Correct switches | Exact accuracy | False abstentions |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for point in history:
        for partition in ("train", "dev"):
            s = point[partition]
            l = s["losses"]
            lines.append(
                f"| {point['step']} | {partition} | "
                f"{l['answerability_bce_all']:.4f} | "
                f"{l['start_ce_answerable']:.4f} | "
                f"{l['end_ce_answerable']:.4f} | "
                f"{s['paired_both_correct']}/{s['paired_total']} | "
                f"{s['correct_switches']}/{s['both_fields_present']} | "
                f"{s['exact_accuracy']:.1%} | "
                f"{s['false_abstentions']} |"
            )
    lines += ["", "## Interpretation", "",
              "Start/end CE are computed on answerable examples only.",
              "Paired accuracy includes correct abstention for missing targets.",
              "Correct switches require both Leader and Location in evidence.",
              "No evaluator, model architecture, loss weights or test partition was changed.",
              "Compare the final TRAIN/DEV values with the original V5G baseline;",
              "a change in performance is not evidence of semantic paraphrase generalization.",
              ""]
    return "\n".join(lines)


def run(checkpoint, output_dir, steps=1200, seed=42, measure_every=120):
    if steps < 1 or measure_every < 1:
        raise ValueError("steps and measure-every must be positive")
    if checkpoint.suffix != ".pt" or checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError("Choose an unused .pt checkpoint filename")
    if output_dir.exists():
        raise FileExistsError("Choose an unused report directory")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = SpanGroundedQA().to(device="cpu", dtype=torch.float32)
    train = read_partition("train")
    dev = read_partition("dev")
    if len(train) != 48 or len(dev) != 48:
        raise ValueError("V5G registered dataset counts changed")
    # Independent sources and fixed answerability ratio are prerequisites.
    if {r["information"] for r in train} & {r["information"] for r in dev}:
        raise ValueError("TRAIN/DEV evidence overlap")
    labels = Counter((r["target_field"], r["answerability"]) for r in train)
    expected_labels = {
        ("leader", "answerable"): 18,
        ("leader", "unanswerable"): 6,
        ("location", "answerable"): 18,
        ("location", "unanswerable"): 6,
    }
    if labels != expected_labels:
        raise ValueError(f"Unexpected training distribution: {labels}")
    prepared_train = prepared_rows("train", model.config.max_span_bytes)
    prepared_dev = prepared_rows("dev", model.config.max_span_bytes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedule = uniform_schedule(len(train), steps, seed)
    exposures = Counter()
    history = []

    def measure(step):
        history.append({"step": step,
                        "train": snapshot(model, train, prepared_train),
                        "dev": snapshot(model, dev, prepared_dev)})
        s = history[-1]
        print(f"Step {step:4d} | TRAIN paired {s['train']['paired_both_correct']}/24"
              f" DEV paired {s['dev']['paired_both_correct']}/24"
              f" | train/dev BCE "
              f"{s['train']['losses']['answerability_bce_all']:.3f}/"
              f"{s['dev']['losses']['answerability_bce_all']:.3f}", flush=True)

    measure(0)
    for step, idx in enumerate(schedule, start=1):
        row, ids, mask, target = prepared_train[idx]
        exposures[f"{row['target_field']}/{row['answerability']}"] += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = span_training_loss(model, ids, mask, target)
        if not torch.isfinite(total).item():
            raise ValueError(f"Nonfinite loss at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % measure_every == 0 or step == steps:
            measure(step)

    training_sha = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    metadata = {
        "experiment": "v5g_leader_location_question_selection",
        "sampling_variant": "uniform_questions_shuffled_epoch",
        "steps": steps,
        "seed": seed,
        "training_sha256": training_sha,
        "training_examples": len(train),
        "supervised_fields": ["leader", "location"],
        "topic_as_training_target": False,
        "answerability_threshold": 0.5,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "checkpoint_role": "v5g_uniform_sampling_ablation",
        "measure_every": measure_every,
    }
    save_span_checkpoint(checkpoint, model, metadata)
    _, restored_meta = load_span_checkpoint(checkpoint)
    if restored_meta != metadata:
        raise ValueError("Saved checkpoint metadata failed round trip")
    report = {
        "experiment": metadata["experiment"], "variant": metadata["sampling_variant"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_sha256": training_sha, "steps": steps, "seed": seed,
        "exposure_counts": dict(sorted(exposures.items())), "measurements": history,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        render(history, exposures, training_sha), encoding="utf-8")
    print(f"Checkpoint: {checkpoint}\nReport: {output_dir / 'summary.md'}")
    print("Reserved EVAL was not accessed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-every", type=int, default=120)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output_dir, args.steps, args.seed, args.measure_every)


if __name__ == "__main__":
    main()
