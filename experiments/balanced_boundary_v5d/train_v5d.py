"""Train an independent V5D checkpoint from random initialization.

Uses registered V3 training + V5D balanced TRAIN only. Never reads DEV or
V5C examples and never overwrites an existing checkpoint.
"""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import torch

from experiments.balanced_boundary_v5d.generate_data_v5d import PATHS, read_split
from experiments.span_extractor_v5.train_span_v5 import (
    mean_training_loss, positive_int, valid_ratio,
)
from grounded_qa.models.span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, prepare_span_input,
    save_span_checkpoint, span_training_loss, training_span_target,
)
from grounded_qa.training.train import (
    EVIDENCE_V3_TRAIN_PATH, class_aware_indices, load_training_examples,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-ratio", type=valid_ratio, default=0.33)
    parser.add_argument("--loss-every", type=positive_int, default=120)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.checkpoint.suffix.lower() != ".pt" or args.checkpoint.exists() or args.checkpoint.is_symlink():
            raise ValueError("Checkpoint must be a new .pt file; overwrite forbidden")
        # V3 loader accepts only its registered TRAIN path.
        v3 = load_training_examples(EVIDENCE_V3_TRAIN_PATH, evidence_experiment=True)
        balanced = read_split("train")
        rows = v3 + balanced
        if len(rows) != 264:
            raise ValueError("Expected 168 V3 + 96 V5D training rows")
        if len({r["information"] for r in rows}) != len(rows):
            raise ValueError("Duplicate evidence between V3 and V5D")
        if len({r["source_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate source IDs")
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        model = SpanGroundedQA().to(device="cpu", dtype=torch.float32)
        prepared = []
        for row in rows:
            source, mask = prepare_span_input(row["information"], row["question"])
            if source.shape[1] > model.config.max_input_tokens:
                raise ValueError("Training input exceeds model limit")
            prepared.append((source, mask, training_span_target(row, source, mask)))
        schedule = class_aware_indices(rows, args.steps, args.missing_ratio)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        initial = mean_training_loss(model, prepared)
        print(f"V5D from scratch: {len(v3)} V3 + {len(balanced)} V5D TRAIN; {args.steps} steps")
        print(f"Before training loss: {initial:.4f}", flush=True)
        started = perf_counter()
        model.train()
        final = initial
        for step, index in enumerate(schedule, start=1):
            source, mask, target = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = span_training_loss(model, source, mask, target)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % args.loss_every == 0 or step == args.steps:
                final = mean_training_loss(model, prepared)
                print(f"Step {step}: mean train loss {final:.4f}", flush=True)
        metadata = {
            "experiment": "v5d_balanced_boundaries",
            "model_variant": "span_v5",
            "steps": args.steps,
            "seed": args.seed,
            "training_file": EVIDENCE_V3_TRAIN_PATH.as_posix(),
            "extra_training_file": PATHS["train"].as_posix(),
            "extra_training_sha256": hashlib.sha256(PATHS["train"].read_bytes()).hexdigest(),
            "example_count": len(rows),
            "prompt_version": "compact_v1",
            "missing_ratio": args.missing_ratio,
            "loss_every": args.loss_every,
            "initial_loss": initial,
            "final_loss": final,
            "answerability_threshold": 0.5,
            "learning_rate": 0.001,
        }
        save_span_checkpoint(args.checkpoint, model, metadata)
        restored, saved = load_span_checkpoint(args.checkpoint)
        if saved != metadata:
            raise ValueError("Checkpoint metadata differs after reload")
        restored.eval()
        model.eval()
        with torch.no_grad():
            source, mask, _ = prepared[0]
            if not all(torch.equal(a, b) for a, b in zip(model(source, mask), restored(source, mask))):
                raise ValueError("Checkpoint reload prediction differs")
        print(f"Training complete in {perf_counter()-started:.1f}s")
        print(f"Checkpoint saved and reload checked: {args.checkpoint}")
        print("Training loss is NOT an evaluation metric; DEV and holdout untouched.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5D training stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
