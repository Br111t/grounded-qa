"""Train V5B using original V3 train plus locked 3/5/6-byte V5B additions.

No evaluation or holdout records are loaded by this trainer.
"""

import argparse
import math
from pathlib import Path
from time import perf_counter

import torch

from generate_span_data_v5b import EXTRA_PATH, read_additions
from span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, prepare_span_input,
    save_span_checkpoint, span_training_loss, training_span_target,
)
from train import EVIDENCE_V3_TRAIN_PATH, class_aware_indices, load_training_examples
from train_span_v5 import mean_training_loss, positive_int, valid_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-data", type=Path, required=True)
    parser.add_argument("--extra-train", type=Path, required=True)
    parser.add_argument("--steps", type=positive_int, default=3000)
    parser.add_argument("--loss-every", type=positive_int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-ratio", type=valid_ratio, default=0.33)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.experiment_data.resolve() != EVIDENCE_V3_TRAIN_PATH.resolve():
            raise ValueError("V5B uses only registered V3 TRAINING data.")
        if args.extra_train.resolve() != EXTRA_PATH.resolve():
            raise ValueError("V5B additions require the locked extra TRAINING path.")
        if args.checkpoint.suffix.lower() != ".pt" or args.checkpoint.exists() or args.checkpoint.is_symlink():
            raise ValueError("Choose a new .pt checkpoint filename; overwriting is forbidden.")
        v3_rows = load_training_examples(args.experiment_data, evidence_experiment=True)
        extra_rows = read_additions("extra_train")
        rows = v3_rows + extra_rows

        if len(rows) != 240:
            raise ValueError("V5B requires 168 V3 + 72 extra training rows.")
        if len({row["source_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate source IDs between V3 and V5B training.")
        if len({row["information"] for row in rows}) != len(rows):
            raise ValueError("Duplicate evidence across V3 and V5B training.")
        if len({row["answer"][8:-1] for row in v3_rows if row["answerability"] == "answerable"}
               & {row["answer"][8:-1] for row in extra_rows}):
            raise ValueError("Extra training names overlap with V3 training names.")

        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        model = SpanGroundedQA().to(device="cpu", dtype=torch.float32)
        prepared = []
        for row in rows:
            source, mask = prepare_span_input(row["information"], row["question"])
            if source.shape[1] > model.config.max_input_tokens:
                raise ValueError("Evidence exceeds max input length.")
            target = training_span_target(row, source, mask)
            prepared.append((source, mask, target))
        schedule = class_aware_indices(rows, args.steps, args.missing_ratio)
        missing_updates = sum(
            rows[i]["answerability"] == "unanswerable" for i in schedule
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("V5B: unchanged encoder-only span architecture, new variable-length training examples.")
        print(f"Original V3 train: {len(v3_rows)}; extra train: {len(extra_rows)}; total: {len(rows)}")
        print("Extra leader-name lengths: 3, 5, 6 (24 rows each); V3 originals: 4.")
        print(f"Seed {args.seed}; updates {args.steps}; batch 1, CPU, one thread.")
        print(f"Balanced sampling: missing {missing_updates}/{args.steps}; answerable {args.steps-missing_updates}/{args.steps}.")
        print("Loss: answerability BCE + start/end cross entropy. No dev/holdout files read.")
        print(f"Checkpoint destination: {args.checkpoint}")
        before = mean_training_loss(model, prepared)
        print(f"Before training: {before:.4f}", flush=True)
        began = perf_counter()
        last = before
        model.train()
        for step, index in enumerate(schedule, start=1):
            source, mask, target = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = span_training_loss(model, source, mask, target)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite loss at step {step}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step % args.loss_every == 0 or step == args.steps:
                last = mean_training_loss(model, prepared)
                print(f"After {step} updates: {last:.4f}", flush=True)

        print(f"Before -> after: {before:.4f} -> {last:.4f}")
        print(f"Training time: {perf_counter()-began:.1f} seconds")
        training = {
            "model_variant": "span_v5",
            "experiment": "v5b_variable_length",
            "steps": args.steps,
            "seed": args.seed,
            "training_file": args.experiment_data.as_posix(),
            "extra_training_file": args.extra_train.as_posix(),
            "example_count": len(rows),
            "prompt_version": "compact_v1",
            "missing_ratio": args.missing_ratio,
            "scheduled_missing_updates": missing_updates,
            "loss_every": args.loss_every,
            "initial_loss": before,
            "final_loss": last,
            "answerability_threshold": 0.5,
            "learning_rate": 0.001,
        }
        save_span_checkpoint(args.checkpoint, model, training)
        restored, saved_info = load_span_checkpoint(args.checkpoint)
        if saved_info != training:
            raise ValueError("Reload changed training metadata.")
        model.eval()
        with torch.no_grad():
            for source, mask, _ in prepared:
                if not all(
                    torch.equal(a, b) for a, b in zip(
                        model(source, mask), restored(source, mask)
                    )
                ):
                    raise ValueError("Reload changed prediction scores.")
        print(f"Checkpoint saved: {args.checkpoint}; reload verified on {len(rows)} train rows.")
        print("No development or final-holdout datasets were used for training.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"V5B training stopped: {error}")
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted; checkpoint is saved only on completion.")
        raise SystemExit(130)
