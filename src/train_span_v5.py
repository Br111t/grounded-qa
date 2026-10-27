"""Train V5 span extractor from random weights on registered V3 TRAINING data only."""

import argparse
import math
from pathlib import Path
from time import perf_counter

import torch

from span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, prepare_span_input,
    save_span_checkpoint, span_training_loss, training_span_target,
)
from train import EVIDENCE_V3_TRAIN_PATH, class_aware_indices, load_training_examples


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def valid_ratio(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number < 1:
        raise argparse.ArgumentTypeError("Ratio must lie strictly between 0 and 1.")
    return number


@torch.no_grad()
def mean_training_loss(model, prepared):
    training = model.training
    model.eval()
    try:
        total = 0.0
        for source, mask, target in prepared:
            total += span_training_loss(model, source, mask, target)[0].item()
        return total / len(prepared)
    finally:
        model.train(training)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-data", required=True, type=Path)
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--loss-every", type=positive_int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-ratio", type=valid_ratio, default=0.33)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.experiment_data.resolve() != EVIDENCE_V3_TRAIN_PATH.resolve():
            raise ValueError("V5 trains only on the registered V3 TRAINING split.")
        if args.checkpoint.suffix.lower() != ".pt" or args.checkpoint.exists() or args.checkpoint.is_symlink():
            raise ValueError("Choose a new .pt checkpoint path; never overwrite an existing checkpoint.")
        rows = load_training_examples(args.experiment_data, evidence_experiment=True)
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        model = SpanGroundedQA().to(device="cpu", dtype=torch.float32)
        prepared = []
        for row in rows:
            source, mask = prepare_span_input(row["information"], row["question"])
            if source.shape[1] > model.config.max_input_tokens:
                raise ValueError("Evidence exceeds model input budget.")
            target = training_span_target(row, source, mask)
            prepared.append((source, mask, target))
        schedule = class_aware_indices(rows, args.steps, args.missing_ratio)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        missing_updates = sum(
            rows[i]["answerability"] == "unanswerable" for i in schedule
        )
        print("Architecture: V5 encoder-only Transformer, span and answerability heads.")
        print(f"Training split: {args.experiment_data}; {len(rows)} validated rows.")
        print(f"Seed: {args.seed}; updates: {args.steps}; batch 1; CPU; one thread.")
        print(f"Missing updates: {missing_updates}; answerable updates: {args.steps-missing_updates}")
        print("Fixed prompt: compact_v1; no pretrained weights; no development or holdout files read.")
        print("Loss: answerability BCE + start/end CE for answerable rows.")
        print("Reported loss is the mean full training objective over training examples.")
        print(f"Checkpoint destination: {args.checkpoint}")
        initial_loss = mean_training_loss(model, prepared)
        print(f"Before training: {initial_loss:.4f}", flush=True)
        started = perf_counter()
        final_loss = initial_loss
        model.train()
        for step, index in enumerate(schedule, start=1):
            source, mask, target = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            total, _, _ = span_training_loss(model, source, mask, target)
            if not torch.isfinite(total).item():
                raise ValueError(f"Non-finite loss at update {step}.")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step % args.loss_every == 0 or step == args.steps:
                final_loss = mean_training_loss(model, prepared)
                print(f"After {step} updates: {final_loss:.4f}", flush=True)
        print(f"Before -> after: {initial_loss:.4f} -> {final_loss:.4f}")
        print(f"Training time: {perf_counter()-started:.1f} seconds")
        print("Training loss is not generated-answer accuracy or holdout accuracy.")

        meta = {
            "model_variant": "span_v5",
            "steps": args.steps,
            "seed": args.seed,
            "training_file": args.experiment_data.as_posix(),
            "example_count": len(rows),
            "prompt_version": "compact_v1",
            "missing_ratio": args.missing_ratio,
            "scheduled_missing_updates": missing_updates,
            "loss_every": args.loss_every,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "answerability_threshold": 0.5,
            "learning_rate": 0.001,
        }
        save_span_checkpoint(args.checkpoint, model, meta)
        restored, restored_meta = load_span_checkpoint(args.checkpoint)
        if restored_meta != meta:
            raise ValueError("Checkpoint metadata changed during reload.")
        model.eval()
        with torch.no_grad():
            for source, mask, _ in prepared:
                old_outputs = model(source, mask)
                new_outputs = restored(source, mask)
                if not all(torch.equal(a, b) for a, b in zip(old_outputs, new_outputs)):
                    raise ValueError("Reload check failed: output scores changed.")
        print(f"Checkpoint saved: {args.checkpoint}")
        print(f"Reload check passed on all {len(rows)} training examples.")
        print("No evaluation/final-holdout datasets were read or changed.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5 training stopped: {error}")
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoint is saved only on completion.")
        raise SystemExit(130)
