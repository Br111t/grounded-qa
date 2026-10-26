"""Train a small CPU practice set: one example per type, or complete source groups."""

import argparse
import math
import re
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from checkpoint import check_destination, load_checkpoint, save_checkpoint
from copy_model import CopyGroundedQA
from evidence_copy_v4b import auxiliary_copy_losses, derive_name_alignment
from evidence_v2_validation import validate_v2_rows
from evidence_v3_validation import validate_v3_rows
from model import TinyGroundedQA
from prompting import PROMPT_VERSIONS, build_prompt
from tokenizer import PAD_ID, VOCAB_SIZE, encode
from validate_data import ANSWERABILITY_VALUES, FALLBACK, validate_data


# This split defines which sources belong to training. An alternate practice
# file may add question wordings for those sources, keeping their facts intact.
TRAIN_PATH = Path("data/training/generated_train.jsonl")
# Deliberately allow only this explicitly registered, synthetic learning experiment.
# This does not authorize arbitrary source files or evaluation/holdout inputs.
EVIDENCE_EXPERIMENT_PATH = Path("data/source/evidence_sensitivity_v1.jsonl")
EVIDENCE_V2_TRAIN_PATH = Path("data/training/evidence_sensitivity_v2_train.jsonl")
EVIDENCE_V3_TRAIN_PATH = Path("data/training/evidence_sensitivity_v3_train.jsonl")
EVIDENCE_QUESTION = "What is the leader of Workshop W0001?"
EVIDENCE_INFORMATION = re.compile(
    r"Fictional Workshop W0001\. (?:Leader: (?P<leader>[A-Za-z]+)\. )?Location: Room Maple\."
)
TokenExample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def load_training_examples(path: Path, *, evidence_experiment: bool = False) -> list[dict]:
    """Validate input; permit new source IDs only for the registered evidence experiment.

    Ordinary --data runs retain the original train-source allowlist.
    The explicit experiment option accepts only the named synthetic source file.
    """
    # Reject arbitrary files BEFORE reading them, including holdout paths.
    if evidence_experiment and path.resolve() not in (
        EVIDENCE_EXPERIMENT_PATH.resolve(), EVIDENCE_V2_TRAIN_PATH.resolve(),
        EVIDENCE_V3_TRAIN_PATH.resolve()
    ):
        raise ValueError("Only the registered evidence-sensitivity training files are allowed.")
    rows, errors = validate_data(path)
    if errors:
        raise ValueError("Training data validation failed:\n  - " + "\n  - ".join(errors))
    if evidence_experiment:
        if path.resolve() == EVIDENCE_V3_TRAIN_PATH.resolve():
            validate_v3_rows(rows, group="train")
            return rows
        if path.resolve() == EVIDENCE_V2_TRAIN_PATH.resolve():
            validate_v2_rows(rows, group="train")
            return rows
        # Preserve all V1 checks and behavior.
        if any(row["question"] != EVIDENCE_QUESTION for row in rows):
            raise ValueError("Evidence-sensitivity examples must use the identical fixed question.")
        if len({row["information"] for row in rows}) != len(rows):
            raise ValueError("Evidence-sensitivity examples must have distinct information variants.")
        for row in rows:
            match = EVIDENCE_INFORMATION.fullmatch(row["information"])
            if match is None:
                raise ValueError(f"Unexpected evidence format: {row['example_id']}")
            leader = match.group("leader")
            expected_answer = f"Leader: {leader}." if leader else FALLBACK
            expected_label = "answerable" if leader else "unanswerable"
            if row["answer"] != expected_answer or row["answerability"] != expected_label:
                raise ValueError(f"Evidence does not support the reference: {row['example_id']}")
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


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("auxiliary loss weights must be finite and nonnegative")
    return number


def class_aware_indices(
    rows: list[dict], steps: int, missing_ratio: float,
) -> list[int]:
    """Return a deterministic class-aware update schedule.

    Round the target number of missing updates once; distribute them across
    training steps and cycle independently through every example in each class.
    The inputs are training examples only, never holdout/evaluation rows.
    """
    if not math.isfinite(missing_ratio) or not 0 < missing_ratio < 1:
        raise ValueError("missing_ratio must be strictly between 0 and 1.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    present = [
        index for index, row in enumerate(rows)
        if row["answerability"] == "answerable"
    ]
    missing = [
        index for index, row in enumerate(rows)
        if row["answerability"] == "unanswerable"
    ]
    if len(present) + len(missing) != len(rows) or not present or not missing:
        raise ValueError("Balanced sampling requires answerable and unanswerable examples only.")

    missing_updates = round(steps * missing_ratio)
    present_seen = 0
    missing_seen = 0
    indices = []
    for step in range(1, steps + 1):
        # Integer arithmetic avoids drift in the ratio across long runs.
        use_missing = (step * missing_updates // steps) > (
            (step - 1) * missing_updates // steps
        )
        if use_missing:
            indices.append(missing[missing_seen % len(missing)])
            missing_seen += 1
        else:
            indices.append(present[present_seen % len(present)])
            present_seen += 1
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument(
        "--data", type=Path,
        help="practice dataset using existing training sources (default: data/training/generated_train.jsonl)",
    )
    dataset.add_argument(
        "--experiment-data", type=Path,
        help="registered V1, V2, or V3 evidence-sensitivity training data; trains on every row",
    )
    parser.add_argument("--steps", type=positive_int, default=60, help="weight updates (default: 60)")
    parser.add_argument(
        "--model-variant", choices=("standard", "pointer_v4"), default="standard",
        help="model architecture (default: standard; pointer_v4 enables learned source copying)",
    )
    parser.add_argument(
        "--alignment-weight", type=nonnegative_float, default=0.0,
        help="V4B pointer attention alignment weight (0 disables; suggested 0.5)",
    )
    parser.add_argument(
        "--copy-gate-weight", type=nonnegative_float, default=0.0,
        help="V4B supervision of copy gate on leader-name bytes (0 disables; suggested 0.5)",
    )
    parser.add_argument(
        "--loss-every", type=positive_int, default=15,
        help="measure full training loss after this many updates (default: 15)",
    )
    parser.add_argument(
        "--sampling", choices=("sequential", "balanced"), default="sequential",
        help="training update schedule (default: sequential, unchanged from V1/V2 baseline)",
    )
    parser.add_argument(
        "--missing-ratio", type=float,
        help="fraction of updates for missing evidence with --sampling balanced (e.g. 0.33)",
    )
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
        if args.sampling == "balanced":
            if not is_evidence_experiment or data_path.resolve() not in (
                EVIDENCE_V2_TRAIN_PATH.resolve(), EVIDENCE_V3_TRAIN_PATH.resolve()
            ):
                raise ValueError("Balanced sampling is restricted to V2/V3 evidence training.")
        elif args.missing_ratio is not None:
            raise ValueError("--missing-ratio requires --sampling balanced.")
        ratio = 0.5 if args.missing_ratio is None else args.missing_ratio
        if args.sampling == "balanced" and not math.isfinite(ratio):
            raise ValueError("--missing-ratio must be finite and strictly between 0 and 1.")
        if args.model_variant == "pointer_v4" and (
            not is_evidence_experiment or data_path.resolve() != EVIDENCE_V3_TRAIN_PATH.resolve()
            or args.prompt_version != "compact_v1"
        ):
            raise ValueError("V4 copy experiments require registered V3 training data and compact_v1.")
        supervised = args.alignment_weight > 0 or args.copy_gate_weight > 0
        if supervised and (
            args.model_variant != "pointer_v4" or not is_evidence_experiment
            or data_path.resolve() != EVIDENCE_V3_TRAIN_PATH.resolve()
            or args.prompt_version != "compact_v1"
        ):
            raise ValueError(
                "V4B auxiliary losses require pointer_v4, registered V3 training, and compact_v1."
            )
        rows = load_training_examples(data_path, evidence_experiment=is_evidence_experiment)
        selected = (
            rows if is_evidence_experiment
            else select_examples(rows, source_id=args.source_id, source_ids=args.source_ids)
        )
        selected_source_ids = sorted({row["source_id"] for row in selected})
        examples = [prepare_example(row, prompt_version=args.prompt_version) for row in selected]
        # Create supervision from validated TRAINING rows only. Check all
        # alignments before a single optimization step. Never inspect eval data.
        name_alignments = (
            [derive_name_alignment(row, prompt_version=args.prompt_version) for row in selected]
            if supervised else None
        )
        update_indices = (
            class_aware_indices(selected, args.steps, ratio)
            if args.sampling == "balanced" else None
        )

        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        model = (
            CopyGroundedQA() if args.model_variant == "pointer_v4" else TinyGroundedQA()
        ).to(device="cpu", dtype=torch.float32)
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
        print(f"Model variant: {args.model_variant}")
        print(
            f"V4B auxiliary weights: attention={args.alignment_weight:g}; "
            f"copy gate={args.copy_gate_weight:g}"
        )
        if supervised:
            print("Name-byte alignment targets from training rows ONLY; inference is unchanged.")
            print("Reported full-dataset losses remain ordinary answer cross-entropy, not total auxiliary loss.")
        print(f"Updates: {args.steps}; learning rate: {args.learning_rate}")
        print(f"Full training loss measurement interval: {args.loss_every} updates")
        if update_indices is None:
            passes, remaining = divmod(args.steps, len(examples))
            print(f"Sampling: sequential; complete passes through selected examples: {passes}; additional updates: {remaining}")
        else:
            missing_updates = sum(
                selected[index]["answerability"] == "unanswerable" for index in update_indices
            )
            print(
                f"Sampling: balanced (class-aware), requested missing ratio {ratio:.3f}; "
                f"scheduled missing updates {missing_updates}/{args.steps} "
                f"({missing_updates / args.steps:.1%}); "
                f"present updates {args.steps - missing_updates}/{args.steps}"
            )
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
            # The default preserves the original sequential, round-robin order.
            example_index = (
                (step - 1) % len(examples)
                if update_indices is None else update_indices[step - 1]
            )
            source, decoder, labels = examples[example_index]

            # 1. Clear gradients left over from the previous update.
            optimizer.zero_grad(set_to_none=True)
            # 2. Predict next-token scores and compare them with correct tokens.
            if supervised:
                logits, components = model.forward_with_components(source, decoder)
            else:
                logits = model(source, decoder)
            answer_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=PAD_ID,
            )
            loss = answer_loss
            if supervised:
                alignment_loss, gate_loss = auxiliary_copy_losses(
                    components, name_alignments[example_index],
                    alignment_weight=args.alignment_weight,
                    gate_weight=args.copy_gate_weight,
                )
                loss = (
                    answer_loss
                    + args.alignment_weight * alignment_loss
                    + args.copy_gate_weight * gate_loss
                )
            if not torch.isfinite(loss).item():
                raise ValueError(f"Loss is not finite at step {step}; stopping before the update.")
            # 3. Calculate how each weight contributed to the loss.
            loss.backward()
            # Limit the combined gradient size and stop if any gradient is invalid.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            # 4. Adjust weights using those gradients.
            optimizer.step()

            if step % args.loss_every == 0 or step == args.steps:
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
                "loss_every": args.loss_every,
                "seed": args.seed,
                "prompt_version": args.prompt_version,
                "model_variant": args.model_variant,
                "alignment_weight": args.alignment_weight,
                "copy_gate_weight": args.copy_gate_weight,
                "training_file": data_path.as_posix(),
                "example_ids": [row["example_id"] for row in selected],
                "source_ids": selected_source_ids,
                "source_count": len(selected_source_ids),
                "selection_mode": selection_mode,
                "sampling": args.sampling,
                "missing_ratio": ratio if args.sampling == "balanced" else None,
                "scheduled_missing_updates": (
                    missing_updates if args.sampling == "balanced" else None
                ),
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
