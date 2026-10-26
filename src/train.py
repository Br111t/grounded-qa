"""Train a small CPU practice set: one example per type, or complete source groups."""

import argparse
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from checkpoint import check_destination, load_checkpoint, save_checkpoint
from model import TinyGroundedQA
from prompting import PROMPT_VERSIONS, build_prompt
from tokenizer import PAD_ID, VOCAB_SIZE, encode
from validate_data import ANSWERABILITY_VALUES, validate_data


# This split defines which sources belong to training. An alternate practice
# file may add question wordings for those sources, keeping their facts intact.
TRAIN_PATH = Path("data/training/generated_train.jsonl")
# Deliberately allow only this explicitly registered, synthetic learning experiment.
# This does not authorize arbitrary source files or evaluation/holdout inputs.
EVIDENCE_EXPERIMENT_PATH = Path("data/source/evidence_sensitivity_v1.jsonl")
EVIDENCE_QUESTION = "What is the leader of Workshop W0001?"
TokenExample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def load_training_examples(path: Path, *, evidence_experiment: bool = False) -> list[dict]:
    """Validate input; permit new source IDs only for the registered evidence experiment.

    Ordinary --data runs retain the original train-source allowlist.
    The explicit experiment option accepts only the named synthetic source file.
    """
    rows, errors = validate_data(path)
    if errors:
        raise ValueError("Training data validation failed:\n  - " + "\n  - ".join(errors))
    if evidence_experiment:
        if path.resolve() != EVIDENCE_EXPERIMENT_PATH.resolve():
            raise ValueError("Only the registered evidence-sensitivity source can be used for this experiment.")
        if any(row["question"] != EVIDENCE_QUESTION for row in rows):
            raise ValueError("Evidence-sensitivity examples must use the identical fixed question.")
        if len({row["information"] for row in rows}) != len(rows):
            raise ValueError("Evidence-sensitivity examples must have distinct information variants.")
        return rows

    if path.resolve() == TRAIN_PATH.resolve():
        return rows

    original_rows, errors = validate_data(TRAIN_PATH)
    if errors:
        raise ValueError("Original training split validation failed:\n  - " + "\n  - ".join(errors))
    training_information = {row["source_id"]: row["information"] for row in original_rows}
    for row in rows:
        source_id = row["source_id"]
        if source_id not in training_information:
            raise ValueError(f"Source ID {source_id!r} is not in the original training split.")
        if row["information"] != training_information[source_id]:
            raise ValueError(f"Information for {source_id!r} differs from the original training split.")
    return rows


def select_examples(
    examples: list[dict], source_id: str | None = None,
    source_ids: list[str] | None = None,
) -> list[dict]:
    """Choose complete source groups, or the first row of each answerability type.

    Source selection preserves file order and uses only the supplied training
    rows. It cannot fetch a missing source from the holdout or generated pool.
    """
    if source_id is not None and source_ids is not None:
        raise ValueError("Choose either a source ID or a list of source IDs.")
    if source_ids is not None:
        if not source_ids or isinstance(source_ids, str):
            raise ValueError("Supply a nonempty list of source IDs.")
        if any(not isinstance(value, str) or not value.strip() for value in source_ids):
            raise ValueError("Source IDs must be nonempty strings.")
        requested = set(source_ids)
        if len(requested) != len(source_ids):
            raise ValueError("Source IDs must not contain duplicates.")
        missing = requested - {row["source_id"] for row in examples}
        if missing:
            raise ValueError("Source IDs not found in the training data: " + ", ".join(sorted(missing)))
        return [row for row in examples if row["source_id"] in requested]
    if source_id is not None:
        selected = [row for row in examples if row["source_id"] == source_id]
        if not selected:
            raise ValueError(f"Source ID {source_id!r} was not found in the training data.")
        return selected

    selected = []
    for label in ANSWERABILITY_VALUES:
        match = next((row for row in examples if row["answerability"] == label), None)
        if match is None:
            raise ValueError(f"Training data needs at least one {label} example.")
        selected.append(match)
    return selected


def prepare_example(example: dict, *, prompt_version: str = "original") -> TokenExample:
    """Make one CPU batch: prompt + EOS, BOS + answer, and answer + EOS."""
    prompt = build_prompt(example["information"], example["question"], version=prompt_version)
    source = torch.tensor([encode(prompt, add_eos=True)], dtype=torch.long)
    decoder = torch.tensor([encode(example["answer"], add_bos=True)], dtype=torch.long)
    labels = torch.tensor([encode(example["answer"], add_eos=True)], dtype=torch.long)
    return source, decoder, labels


def measure_loss(model: TinyGroundedQA, examples: list[TokenExample]) -> float:
    """Measure mean loss per target token on the practice examples, with dropout off.

    This uses training examples, not the holdout. Summing first means that each
    target token has equal weight even when answers have different lengths.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for source, decoder, labels in examples:
                logits = model(source, decoder)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
                    ignore_index=PAD_ID, reduction="sum",
                )
                if not torch.isfinite(loss).item():
                    raise ValueError("Measured loss is not finite; stopping the experiment.")
                total_loss += loss.item()
                total_tokens += labels.ne(PAD_ID).sum().item()
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Cannot measure loss without target tokens.")
    return total_loss / total_tokens


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("steps must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("learning rate must be positive and finite")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument(
        "--data", type=Path,
        help="practice dataset using existing training sources (default: data/training/generated_train.jsonl)",
    )
    dataset.add_argument(
        "--experiment-data", type=Path,
        help="explicit, registered evidence-sensitivity dataset; trains on every row",
    )
    parser.add_argument("--steps", type=positive_int, default=60, help="weight updates (default: 60)")
    parser.add_argument(
        "--learning-rate", type=positive_float, default=0.001,
        help="optimizer adjustment rate (default: 0.001)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--prompt-version", choices=PROMPT_VERSIONS, default="original",
        help="versioned prompt format used for training (default: original)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--source-id", help="train on every question for this training source (default: first row of each type)",
    )
    selection.add_argument(
        "--source-ids", nargs="+", metavar="ID",
        help="train on every question for these training sources, in file order",
    )
    parser.add_argument("--checkpoint", type=Path, help="save to a new .pt file after training (default: no save)")
    args = parser.parse_args()

    try:
        if args.checkpoint is not None:
            check_destination(args.checkpoint)
        data_path = args.experiment_data or args.data or TRAIN_PATH
        is_evidence_experiment = args.experiment_data is not None
        if is_evidence_experiment and (args.source_id is not None or args.source_ids is not None):
            raise ValueError("Evidence experiment uses all examples; omit --source-id and --source-ids.")
        rows = load_training_examples(data_path, evidence_experiment=is_evidence_experiment)
        selected = (
            rows if is_evidence_experiment
            else select_examples(rows, source_id=args.source_id, source_ids=args.source_ids)
        )
        selected_source_ids = sorted({row["source_id"] for row in selected})
        examples = [prepare_example(row, prompt_version=args.prompt_version) for row in selected]

        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        model = TinyGroundedQA().to(device="cpu", dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print(f"Training file: {data_path}")
        if is_evidence_experiment:
            selection = "all rows from the registered evidence-sensitivity experiment"
            selection_mode = "evidence_experiment_all"
        elif args.source_ids is not None:
            selection = "all questions from " + ", ".join(args.source_ids)
            selection_mode = "source_groups"
        elif args.source_id is not None:
            selection = f"all questions from {args.source_id}"
            selection_mode = "source_group"
        else:
            selection = "first example of each answerability type"
            selection_mode = "first_per_type"
        print(f"Selection: {selection}")
        print(f"Selected {len(selected)} of {len(rows)} training examples.")
        print(f"Source groups: {len(selected_source_ids)}")
        print("Practice examples (cycled in this order):")
        for row in selected:
            print(f"  {row['example_id']} ({row['answerability']})")
        print(f"CPU; float32; one thread; batch size 1; seed {args.seed}")
        print(f"Prompt version: {args.prompt_version}")
        print(f"Updates: {args.steps}; learning rate: {args.learning_rate}")
        passes, remaining = divmod(args.steps, len(examples))
        print(f"Complete passes through selected examples: {passes}; additional updates: {remaining}")
        print("Starting from fresh random weights.")
        if args.checkpoint is None:
            print("This practice run saves no checkpoint.")
        else:
            print(f"Checkpoint destination: {args.checkpoint}")
        print(f"Reported losses use these same {len(examples)} training examples with dropout off.")

        started = perf_counter()
        # Check that every selected example fits the model before updating.
        initial_loss = measure_loss(model, examples)
        print(f"Before training: {initial_loss:.4f}", flush=True)
        model.train()  # Enable training behavior, including dropout.

        for step in range(1, args.steps + 1):
            # Visit each selected example in order, then repeat the sequence.
            source, decoder, labels = examples[(step - 1) % len(examples)]

            # 1. Clear gradients left over from the previous update.
            optimizer.zero_grad(set_to_none=True)
            # 2. Predict next-token scores and compare them with correct tokens.
            logits = model(source, decoder)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=PAD_ID,
            )
            if not torch.isfinite(loss).item():
                raise ValueError(f"Loss is not finite at step {step}; stopping before the update.")
            # 3. Calculate how each weight contributed to the loss.
            loss.backward()
            # Limit the combined gradient size and stop if any gradient is invalid.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            # 4. Adjust weights using those gradients.
            optimizer.step()

            if step % 15 == 0 or step == args.steps:
                final_loss = measure_loss(model, examples)
                print(f"After {step:>3} updates: {final_loss:.4f}", flush=True)

        print(f"Before -> after: {initial_loss:.4f} -> {final_loss:.4f}")
        print(f"Experiment time: {perf_counter() - started:.1f} seconds (excluding package imports/setup)")
        if final_loss < initial_loss:
            print("Loss decreased on the practice examples.")
        else:
            print("Loss did not decrease; inspect the settings before extending the run.")
        print("This measures fitting these examples, not generated-answer or holdout accuracy.")
        print("The holdout was not read. No dataset was changed.")
        if is_evidence_experiment:
            print("All evidence variants are TRAINING examples; this run does not measure generalization.")
        if args.checkpoint is None:
            print("Learned weights exist only in memory for this run; no checkpoint was saved.")
        else:
            training_info = {
                "steps": args.steps,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "prompt_version": args.prompt_version,
                "training_file": data_path.as_posix(),
                "example_ids": [row["example_id"] for row in selected],
                "source_ids": selected_source_ids,
                "source_count": len(selected_source_ids),
                "selection_mode": selection_mode,
                "example_count": len(selected),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
            }
            save_checkpoint(args.checkpoint, model, training_info)
            print(f"Checkpoint saved: {args.checkpoint}", flush=True)
            restored_model, _ = load_checkpoint(args.checkpoint)
            model.eval()
            with torch.no_grad():
                for source, decoder, _ in examples:
                    if not torch.equal(model(source, decoder), restored_model(source, decoder)):
                        raise ValueError("Reload check failed: prediction scores changed after reloading.")
            print(f"Reload check passed: identical scores on all {len(examples)} practice examples.")
            print("Saved model weights and settings; optimizer history is not included.")
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        print(f"Training experiment stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPractice run interrupted. A checkpoint is saved only after training completes.")
        raise SystemExit(130)
