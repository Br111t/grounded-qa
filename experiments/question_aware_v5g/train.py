"""V5G train-from-scratch on paired Leader/Location supervision only.

Model architecture remains span_v5; V5G is a new TRAINING experiment.
No existing checkpoints are modified and no evaluation partition is read.
"""
import argparse
import hashlib
from pathlib import Path

import torch

from experiments.question_aware_v5g.data_design import DATA, read_partition
from grounded_qa.models.span_model_v5 import (
    SOURCE_PREFIX, SpanGroundedQA, load_span_checkpoint, prepare_span_input,
    save_span_checkpoint, span_training_loss,
)
from grounded_qa.tokenization.tokenizer import decode, encode


def target_for(row, token_ids, mask, max_span_bytes=16):
    """Question-conditioned gold target derived from TRAIN-only row."""
    if row["target_field"] not in ("leader", "location"):
        raise ValueError("V5G trains only on Leader/Location")
    if row["answerability"] == "unanswerable":
        if row["answer"] != "Insufficient information." or row["expected_value"] is not None:
            raise ValueError("Inconsistent missing-target training reference")
        return (0.0, None, None)
    field = row["target_field"].title()
    value = row["expected_value"]
    marker = f"{field}: {value}."
    information = row["information"]
    if not value or information.count(marker) != 1 or row["answer"] != marker:
        raise ValueError("Training reference is not uniquely evidence grounded")
    offset = information.index(marker) + len(field) + 2
    start = 1 + len(encode(SOURCE_PREFIX + information[:offset]))
    length = len(encode(value))
    end = start + length - 1
    if length > max_span_bytes or not mask[0, start:end + 1].all().item():
        raise ValueError("Gold evidence span out of bounds")
    if decode(token_ids[0, start:end + 1].tolist()) != value:
        raise ValueError("Gold span does not match tokenized evidence")
    return (1.0, start, end)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--log-every", type=int, default=120)
    args = ap.parse_args()
    if args.steps < 1 or args.log_every < 1:
        ap.error("steps and log-every must be positive")
    if args.checkpoint.exists() or args.checkpoint.suffix.lower() != ".pt":
        ap.error("Choose a new unused .pt checkpoint path")
    rows = read_partition("train")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = SpanGroundedQA()
    prepared = []
    for row in rows:
        ids, mask = prepare_span_input(row["information"], row["question"])
        if ids.shape[1] > model.config.max_input_tokens:
            raise ValueError("Training prompt too long")
        prepared.append((ids, mask, target_for(row, ids, mask, model.config.max_span_bytes)))
    # Deterministic balanced round-robin across both field types and
    # answerability. Each epoch visits every training row once.
    groups = {}
    for i, row in enumerate(rows):
        key = (row["target_field"], row["answerability"])
        groups.setdefault(key, []).append(i)
    if set(groups) != {
        ("leader", "answerable"), ("leader", "unanswerable"),
        ("location", "answerable"), ("location", "unanswerable")
    }:
        raise ValueError("Missing training categories")
    keys = sorted(groups)
    rng = __import__("random").Random(args.seed)
    for group in groups.values():
        rng.shuffle(group)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"V5G TRAIN: {len(rows)} questions, 24 paired evidence records, no eval data")
    for step in range(args.steps):
        key = keys[step % len(keys)]
        candidates = groups[key]
        index = candidates[(step // len(keys)) % len(candidates)]
        ids, mask, target = prepared[index]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = span_training_loss(model, ids, mask, target)
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            print(f"Step {step+1}: train loss {loss.item():.4f}", flush=True)
    path = DATA / "train.jsonl"
    meta = {
        "experiment": "v5g_leader_location_question_selection",
        "steps": args.steps, "seed": args.seed,
        "training_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_examples": len(rows),
        "supervised_fields": ["leader", "location"],
        "topic_as_training_target": False,
        "answerability_threshold": 0.5,
        "optimizer": "Adam", "learning_rate": 0.001,
        "checkpoint_role": "new_independent_v5g_weights",
    }
    save_span_checkpoint(args.checkpoint, model, meta)
    _, loaded = load_span_checkpoint(args.checkpoint)
    if loaded != meta:
        raise ValueError("Checkpoint metadata round trip mismatch")
    print(f"Saved: {args.checkpoint}")
    print("No DEV, paraphrase, unrelated, or Topic evaluation has been performed.")


if __name__ == "__main__":
    main()
