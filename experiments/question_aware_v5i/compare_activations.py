"""Controlled V5I fusion-activation comparison: tanh, GELU, linear.

TRAIN/DEV only. New independent checkpoints, unchanged encoder/data/loss/
sampling/decoder. Each variant starts from the same seeded initialization.
Existing V5I files and reserved EVAL remain untouched.
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.diagnose_loss_components import prepared_rows
from experiments.question_aware_v5g.train_uniform_sampling import uniform_schedule
from experiments.question_aware_v5i.train_question_conditioned import snapshot, train_loss
from grounded_qa.models.span_model_v5i import QuestionConditionedSpanQA
from grounded_qa.tokenization.tokenizer import BOS_ID, BYTE_OFFSET, PAD_ID, VOCAB_SIZE

VARIANTS = ("tanh", "gelu", "linear")
EXPERIMENT = "v5i_activation_comparison"
ROOT = Path("experiments/question_aware_v5i")


class ActivationSpanQA(QuestionConditionedSpanQA):
    """Identical learned parameter structure; only fusion activation varies."""

    def __init__(self, activation):
        if activation not in VARIANTS:
            raise ValueError("Unknown activation")
        super().__init__()
        self.activation = activation

    def forward(self, token_ids, candidate_mask):
        # Mirror V5I's input-validation and encoder pathway.
        if token_ids.ndim != 2 or not all(token_ids.shape):
            raise ValueError("Token IDs must be nonempty [batch,length]")
        if token_ids.dtype != torch.long or token_ids.shape[1] > self.config.max_input_tokens:
            raise ValueError("Invalid token dtype or length")
        if candidate_mask.shape != token_ids.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("Invalid evidence candidate mask")
        if token_ids.device != candidate_mask.device:
            raise ValueError("Token/mask device mismatch")
        if not token_ids[:, 0].eq(BOS_ID).all().item():
            raise ValueError("BOS expected")
        if ((token_ids < 0) | (token_ids >= VOCAB_SIZE)).any().item():
            raise ValueError("Invalid token ID")
        if (~token_ids.ge(BYTE_OFFSET) & candidate_mask).any().item():
            raise ValueError("Candidate mask includes nonbyte token")
        padding = token_ids.eq(PAD_ID)
        if (padding[:, :-1] & ~padding[:, 1:]).any().item():
            raise ValueError("Only right padding supported")
        qmask = self.question_mask(token_ids, candidate_mask)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        embedded = self.embedding(token_ids) * math.sqrt(self.config.d_model) + self.positions(positions)
        hidden = self.encoder(
            embedded, src_key_padding_mask=padding if padding.any().item() else None
        )
        question = (hidden * qmask.unsqueeze(-1)).sum(dim=1) / qmask.sum(dim=1, keepdim=True)
        combined = torch.cat((hidden, question.unsqueeze(1).expand_as(hidden)), dim=-1)
        z = self.question_fusion(combined)
        if self.activation == "tanh":
            adjustment = torch.tanh(z)
        elif self.activation == "gelu":
            adjustment = F.gelu(z, approximate="none")
        else:
            adjustment = z
        conditioned = hidden + adjustment
        start = self.start_head(conditioned).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        end = self.end_head(conditioned).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        answerability = self.answerability_head(conditioned[:, 0]).squeeze(-1)
        return start, end, answerability


def save_checkpoint(path, model, metadata):
    if path.exists() or path.is_symlink() or path.suffix.lower() != ".pt":
        raise FileExistsError(f"Checkpoint must be new .pt file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT,
        "activation": model.activation,
        "config": vars(model.config),
        "metadata": metadata,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    # Exclusive creation prevents accidental overwrite of the original V5I.
    with path.open("xb") as f:
        torch.save(payload, f)


def train_variant(name, checkpoint, report_dir, train, dev, schedule, *, steps, measure_every, seed):
    if checkpoint.exists() or checkpoint.is_symlink() or report_dir.exists() or report_dir.is_symlink():
        raise FileExistsError(f"Choose unused checkpoint/report paths for {name}")
    torch.manual_seed(seed)
    model = ActivationSpanQA(name).to(device="cpu", dtype=torch.float32)
    prepared_train = prepared_rows("train", model.config.max_span_bytes)
    prepared_dev = prepared_rows("dev", model.config.max_span_bytes)
    candidates = [
        complete_field_spans(row["information"], model.config.max_span_bytes)
        for row in train
    ]
    for row, spans in zip(train, candidates):
        fields = {c["field"].lower() for c in spans}
        if row["answerability"] == "answerable" and row["target_field"] not in fields:
            raise ValueError("Missing target field")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, exposures = [], Counter()
    eligible_updates, optimization_seconds = 0, 0.0
    started = time.perf_counter()

    def measure(step):
        record = {
            "step": step,
            "train": snapshot(model, train, prepared_train),
            "dev": snapshot(model, dev, prepared_dev),
        }
        history.append(record)
        v = record["dev"]
        print(
            f"{name:6s} step {step:4d} | DEV {v['correct']}/48 "
            f"paired {v['paired_both_correct']}/24 "
            f"switches {v['correct_switches']}/12 "
            f"unsupported {v['unsupported_answers']}/12", flush=True
        )

    measure(0)
    for step, index in enumerate(schedule, start=1):
        row, ids, mask, target = prepared_train[index]
        exposures[f"{row['target_field']}/{row['answerability']}"] += 1
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, rank, eligible = train_loss(
            model, ids, mask, target, candidates[index], row["target_field"],
            margin=1.0, rank_weight=0.5,
        )
        if not torch.isfinite(total).item():
            raise ValueError(f"Nonfinite loss at step {step} ({name})")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimization_seconds += time.perf_counter() - t0
        eligible_updates += int(eligible)
        if step % measure_every == 0 or step == steps:
            measure(step)

    metadata = {
        "experiment_variant": EXPERIMENT,
        "activation": name,
        "seed": seed, "steps": steps, "measure_every": measure_every,
        "training_sha256": hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256((DATA / "dev.jsonl").read_bytes()).hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam", "learning_rate": 0.001,
        "rank_margin": 1.0, "rank_weight": 0.5,
        "dropout": model.config.dropout, "gradient_clip_norm": 1.0,
        "uniform_schedule": True,
    }
    elapsed = time.perf_counter() - started
    save_checkpoint(checkpoint, model, metadata)
    report = {
        "metadata": metadata,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "optimization_seconds": optimization_seconds,
        "elapsed_seconds": elapsed,
        "ranking_eligible_updates": eligible_updates,
        "exposure_counts": dict(sorted(exposures.items())),
        "measurements": history,
        "reserved_eval_accessed": False,
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def run(output_dir, checkpoint_dir, *, steps=1200, seed=42, measure_every=120):
    if steps < 1 or measure_every < 1:
        raise ValueError("Invalid training steps or measurement interval")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Comparison output directory already exists")
    if not checkpoint_dir.is_dir():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {name: checkpoint_dir / f"v5i_activation_{name}_seed{seed}_steps{steps}.pt"
                   for name in VARIANTS}
    for name in VARIANTS:
        if checkpoints[name].exists() or checkpoints[name].is_symlink():
            raise FileExistsError(f"Existing checkpoint: {checkpoints[name]}")
    torch.set_num_threads(1)
    train, dev = read_partition("train"), read_partition("dev")
    if len(train) != 48 or len(dev) != 48:
        raise ValueError("Unexpected registered dataset length")
    if {r["information"] for r in train} & {r["information"] for r in dev}:
        raise ValueError("TRAIN/DEV source overlap")
    counts = Counter((r["target_field"], r["answerability"]) for r in train)
    if counts != {
        ("leader", "answerable"): 18, ("leader", "unanswerable"): 6,
        ("location", "answerable"): 18, ("location", "unanswerable"): 6,
    }:
        raise ValueError("Unexpected training target distribution")
    schedule = uniform_schedule(len(train), steps, seed)
    results = {}
    for name in VARIANTS:
        results[name] = train_variant(
            name, checkpoints[name], output_dir / name,
            train, dev, schedule, steps=steps, seed=seed, measure_every=measure_every
        )

    lines = [
        "# V5I activation comparison: tanh vs GELU vs linear", "",
        "Matched architecture, random initialization seed, TRAIN/DEV, schedule, optimizer,",
        "objective, decoder and evaluation cadence. Only activation changes.",
        "One fixed seed; results are exploratory, not a multiple-seed comparison.",
        "No reserved EVAL accessed. New checkpoints are independent of original V5I.", "",
        "| Variant | DEV exact | Paired both-correct | Correct switches | Unsupported | False abstentions | TRAIN exact | Optimization s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in VARIANTS:
        r = results[name]
        end = r["measurements"][-1]
        v, t = end["dev"], end["train"]
        lines.append(
            f"| {name} | {v['correct']}/48 | {v['paired_both_correct']}/24 | "
            f"{v['correct_switches']}/12 | {v['unsupported_answers']}/12 | "
            f"{v['false_abstentions']} | {t['correct']}/48 | "
            f"{r['optimization_seconds']:.3f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "- The linear fusion still has question-dependent encoder hidden states; only its direct",
        "  additive pooled-question term cancels in within-encoder token rankings.",
        "- Relative scores and paired-switch metrics take precedence over raw logit magnitude.",
        "- Record Brier/answerability, losses and field confusion in per-variant metrics.json.",
        "- DEV is exploratory; do not select an EVAL winner without preregistration.",
        "- Compare runtimes only with similar machine load. One seed is not statistical evidence.",
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Reserved EVAL was not accessed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-every", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    run(args.output_dir, args.checkpoint_dir,
        steps=args.steps, seed=args.seed, measure_every=args.measure_every)


if __name__ == "__main__":
    main()
