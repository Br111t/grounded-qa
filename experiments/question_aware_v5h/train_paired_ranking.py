"""V5H: controlled paired-field ranking ablation on the V5G uniform baseline.

Same model, train/dev data, shuffled-question schedule, optimizer, seed and
complete-field decoder as V5G uniform. Only the TRAIN loss gains a ranking
penalty when both Leader and Location are present. Reserved EVAL is untouched.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.diagnose_loss_components import (
    aggregate, prepared_rows, score,
)
from experiments.question_aware_v5g.evaluate import enriched, predict, summarize
from experiments.question_aware_v5g.train_uniform_sampling import uniform_schedule
from grounded_qa.models.span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, save_span_checkpoint,
)

ROOT = Path("experiments/question_aware_v5h")
FIELDS = frozenset(("leader", "location"))


def paired_rank_penalty(start_logits, end_logits, candidates, target_field, margin=1.0):
    """Differentiable hinge on complete-field joint logits.

    Only rows with both fields are eligible. Grounded candidate positions are
    parsed from evidence; gold target_field is used for TRAIN supervision only.
    """
    if target_field not in FIELDS:
        raise ValueError("Ranking target must be Leader or Location")
    if margin <= 0:
        raise ValueError("Ranking margin must be positive")
    by_field = {}
    for candidate in candidates:
        field = candidate["field"].lower()
        if field in FIELDS:
            if field in by_field:
                raise ValueError("Ambiguous duplicate candidate field")
            by_field[field] = candidate
    if set(by_field) != FIELDS:
        return start_logits.sum() * 0.0, False
    for candidate in by_field.values():
        s, e = candidate["start"], candidate["end"]
        if not 0 <= s <= e < start_logits.shape[1]:
            raise ValueError("Candidate span outside logits")
    def joint(field):
        c = by_field[field]
        return start_logits[0, c["start"]] + end_logits[0, c["end"]]
    rival = "location" if target_field == "leader" else "leader"
    return F.relu(margin - (joint(target_field) - joint(rival))), True


def train_loss(model, ids, mask, target, candidates, target_field,
               *, margin=1.0, rank_weight=0.5):
    """Exactly V5G BCE + answerable-only mean CE, plus V5H ranking term."""
    if rank_weight < 0:
        raise ValueError("rank_weight must be nonnegative")
    starts, ends, answers = model(ids, mask)
    answerable, start, end = target
    classification = F.binary_cross_entropy_with_logits(
        answers, answers.new_tensor([answerable])
    )
    if answerable:
        if start is None or end is None:
            raise ValueError("Answerable row missing gold span")
        extraction = (
            F.cross_entropy(starts, torch.tensor([start], device=starts.device))
            + F.cross_entropy(ends, torch.tensor([end], device=ends.device))
        ) / 2
        rank, eligible = paired_rank_penalty(
            starts, ends, candidates, target_field, margin
        )
    else:
        if start is not None or end is not None:
            raise ValueError("Unanswerable row has gold span")
        extraction = classification * 0.0
        rank, eligible = classification * 0.0, False
    return classification + extraction + rank_weight * rank, rank, eligible


@torch.no_grad()
def snapshot(model, rows, prepared):
    model.eval()
    losses = aggregate(score(model, prepared))
    outcomes = [enriched(row, predict(model, row)) for row in rows]
    metrics = summarize(outcomes)
    sensitivity = metrics["paired"]["question_sensitivity"]
    def avg(values):
        return sum(values) / len(values) if values else None
    return {
        "losses": losses,
        "exact_accuracy": metrics["canonical"]["accuracy"],
        "correct": metrics["canonical"]["correct"],
        "leader_correct": metrics["leader_canonical"]["correct"],
        "location_correct": metrics["location_canonical"]["correct"],
        "false_abstentions": metrics["canonical"]["false_abstentions"],
        "unsupported_answers": metrics["canonical"]["unsupported_answers"],
        "brier_score": metrics["canonical"]["brier_score"],
        "field_confusion": metrics["canonical"]["field_confusion"],
        "paired_both_correct": metrics["paired"]["both_correct"],
        "paired_total": metrics["paired"]["count"],
        "correct_switches": metrics["paired"]["correct_switches"],
        "both_fields_present": metrics["paired"]["both_fields_present"],
        "selection_switches": metrics["paired"]["selection_switches"],
        "mean_leader_question_margin": avg([
            r["leader_advantage_leader_question"] for r in sensitivity
        ]),
        "mean_location_question_margin": avg([
            r["location_advantage_location_question"] for r in sensitivity
        ]),
        "answerable_accuracy": sum(
            r["correct"] for r in outcomes if r["expected_value"] is not None
        ) / sum(r["expected_value"] is not None for r in outcomes),
        "unanswerable_accuracy": sum(
            r["correct"] for r in outcomes if r["expected_value"] is None
        ) / sum(r["expected_value"] is None for r in outcomes),
    }


def baseline_final(path, training_sha, steps, seed):
    if path is None:
        return None
    base = json.loads(path.read_text(encoding="utf-8"))
    if (base.get("variant") != "uniform_questions_shuffled_epoch"
            or base.get("training_sha256") != training_sha
            or base.get("steps") != steps or base.get("seed") != seed):
        raise ValueError("Baseline does not match V5G uniform data, steps and seed")
    points = base.get("measurements", [])
    if not points or points[-1]["step"] != steps:
        raise ValueError("Baseline does not include final step")
    return {"path": str(path), "train": points[-1]["train"],
            "dev": points[-1]["dev"]}


def render(history, exposures, meta, baseline, runtime):
    lines = [
        "# V5H paired-ranking loss — controlled ablation", "",
        "Same V5G uniform train/dev, shuffled schedule, optimizer, model and decoder.",
        "Only training loss changes: BCE + (start CE + end CE)/2 + ",
        "rank_weight * max(0, margin - (requested joint score - rival joint score)).",
        "Ranking applies only to answerable rows containing BOTH Leader and Location.",
        "Topic is never a ranking candidate. DEV is diagnostic; reserved EVAL unused.",
        "",
        "## Configuration", "",
        f"- Seed: {meta['seed']}; updates: {meta['steps']}; Adam LR: {meta['learning_rate']}",
        f"- Ranking weight: {meta['rank_weight']}; margin: {meta['rank_margin']}",
        f"- Parameters: {meta['parameter_count']}; train SHA256: {meta['training_sha256']}",
        f"- Total elapsed seconds (includes diagnostics): {runtime['elapsed_seconds']:.2f}",
        f"- Optimization seconds (excludes diagnostics): {runtime['optimization_seconds']:.2f}",
        f"- Ranking-eligible updates: {runtime['ranking_eligible_updates']}",
        "", "## Exposures", "",
        "| Field / answerability | Updates |", "| --- | ---: |",
    ]
    for key, count in sorted(exposures.items()):
        lines.append(f"| {key} | {count} |")
    lines += [
        "", "## Whole-split DEV/ TRAIN diagnostics (dropout off)", "",
        "| Step | Split | BCE | Start CE | End CE | Exact | Paired | Switches | False abstain | Unsupported |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in history:
        for split in ("train", "dev"):
            v = point[split]
            losses = v["losses"]
            lines.append(
                f"| {point['step']} | {split} | "
                f"{losses['answerability_bce_all']:.3f} | "
                f"{losses['start_ce_answerable']:.3f} | "
                f"{losses['end_ce_answerable']:.3f} | "
                f"{v['correct']}/48 | "
                f"{v['paired_both_correct']}/24 | "
                f"{v['correct_switches']}/12 | "
                f"{v['false_abstentions']} | {v['unsupported_answers']} |"
            )
    if baseline is not None:
        lines += [
            "", "## Final DEV versus existing V5G uniform baseline", "",
            "| Metric | V5G uniform | V5H ranking |",
            "| --- | ---: | ---: |",
        ]
        left, right = baseline["dev"], history[-1]["dev"]
        for k in ("correct", "leader_correct", "location_correct",
                  "paired_both_correct", "correct_switches",
                  "false_abstentions", "unsupported_answers", "brier_score"):
            # Earlier V5G uniform snapshots did not record Brier score.
            # Missing baseline metrics must be labeled, not invented.
            lines.append(f"| {k} | {left.get(k, 'not recorded')} | {right[k]} |")
    lines += [
        "", "## Interpretation guardrails", "",
        "- Compare DEV paired both-correct, correct switches, and BOTH abstention error types.",
        "- Higher field margins alone do not establish accuracy or calibration.",
        "- The checkpoint is new. Existing V5G checkpoints and reports are not changed.",
        "- This does not validate paraphrases or Topic transfer.",
        "- Model parameters are unchanged; training FLOPs may increase modestly.",
        "- Wall time includes machine load and diagnostic cadence; V5G's prior report",
        "  lacks timing, so do not claim measured speed improvements over that run.",
        "- Do not generate reserved EVAL or use any prior freeze for this checkpoint.",
        "",
    ]
    return "\n".join(lines)


def run(checkpoint, output_dir, *, steps=1200, seed=42,
        measure_every=120, margin=1.0, rank_weight=0.5,
        baseline_metrics=None):
    if steps < 1 or measure_every < 1 or margin <= 0 or rank_weight <= 0:
        raise ValueError("Invalid steps, measurement cadence, margin or rank weight")
    if (checkpoint.suffix.lower() != ".pt" or checkpoint.exists()
            or checkpoint.is_symlink()):
        raise FileExistsError("Use a new, unused .pt checkpoint path")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Use a new, unused report directory")
    train, dev = read_partition("train"), read_partition("dev")
    if len(train) != 48 or len(dev) != 48:
        raise ValueError("Registered train/dev dataset counts changed")
    if {r["information"] for r in train} & {r["information"] for r in dev}:
        raise ValueError("TRAIN and DEV evidence overlap")
    counts = Counter((r["target_field"], r["answerability"]) for r in train)
    if counts != {
        ("leader", "answerable"): 18, ("leader", "unanswerable"): 6,
        ("location", "answerable"): 18, ("location", "unanswerable"): 6,
    }:
        raise ValueError("Unexpected training target distribution")
    training_sha = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    baseline = baseline_final(baseline_metrics, training_sha, steps, seed)

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = SpanGroundedQA().to(device="cpu", dtype=torch.float32)
    prepared_train = prepared_rows("train", model.config.max_span_bytes)
    prepared_dev = prepared_rows("dev", model.config.max_span_bytes)
    candidates = [
        complete_field_spans(r["information"], model.config.max_span_bytes)
        for r in train
    ]
    for row, fields in zip(train, candidates):
        labels = {c["field"].lower() for c in fields}
        if row["answerability"] == "answerable" and row["target_field"] not in labels:
            raise ValueError("Answerable gold field missing from candidates")
    schedule = uniform_schedule(len(train), steps, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    exposures = Counter()
    eligible_updates = 0
    optimization_seconds = 0.0
    start_wall = time.perf_counter()

    def measure(step):
        record = {
            "step": step,
            "elapsed_seconds": time.perf_counter() - start_wall,
            "optimization_seconds": optimization_seconds,
            "train": snapshot(model, train, prepared_train),
            "dev": snapshot(model, dev, prepared_dev),
        }
        history.append(record)
        v = record["dev"]
        print(
            f"Step {step:4d} | DEV exact {v['correct']}/48 "
            f"paired {v['paired_both_correct']}/24 "
            f"switch {v['correct_switches']}/12 "
            f"unsupported {v['unsupported_answers']}/12",
            flush=True,
        )

    measure(0)
    for step, idx in enumerate(schedule, start=1):
        row, ids, mask, target = prepared_train[idx]
        exposures[f"{row['target_field']}/{row['answerability']}"] += 1
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, rank, eligible = train_loss(
            model, ids, mask, target, candidates[idx], row["target_field"],
            margin=margin, rank_weight=rank_weight,
        )
        if not torch.isfinite(total).item():
            raise ValueError(f"Nonfinite loss at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimization_seconds += time.perf_counter() - t0
        eligible_updates += int(eligible)
        if step % measure_every == 0 or step == steps:
            measure(step)

    runtime = {
        "elapsed_seconds": time.perf_counter() - start_wall,
        "optimization_seconds": optimization_seconds,
        "ranking_eligible_updates": eligible_updates,
    }
    metadata = {
        "experiment": "v5g_leader_location_question_selection",
        "experiment_variant": "v5h_paired_ranking",
        "checkpoint_role": "v5h_independent_paired_ranking_weights",
        "steps": steps, "seed": seed, "measure_every": measure_every,
        "training_sha256": training_sha, "training_examples": len(train),
        "supervised_fields": ["leader", "location"],
        "topic_as_training_target": False,
        "sampling_variant": "uniform_questions_shuffled_epoch",
        "answerability_threshold": 0.5,
        "optimizer": "Adam", "learning_rate": 0.001,
        "rank_margin": margin, "rank_weight": rank_weight,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    save_span_checkpoint(checkpoint, model, metadata)
    _, restored_meta = load_span_checkpoint(checkpoint)
    if restored_meta != metadata:
        raise ValueError("Saved V5H checkpoint metadata mismatch")
    report = {
        "metadata": metadata,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "exposure_counts": dict(sorted(exposures.items())),
        "runtime": runtime,
        "baseline": baseline, "measurements": history,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render(history, exposures, metadata, baseline, runtime),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Report: {output_dir / 'summary.md'}")
    print("Reserved EVAL was not accessed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-every", type=int, default=120)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    args = parser.parse_args()
    run(
        args.checkpoint, args.output_dir, steps=args.steps, seed=args.seed,
        measure_every=args.measure_every, margin=args.margin,
        rank_weight=args.rank_weight, baseline_metrics=args.baseline_metrics,
    )


if __name__ == "__main__":
    main()
