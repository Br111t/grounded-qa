"""Generate answers from saved weights, using only information and a question."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from model import TinyGroundedQA
from prompting import build_prompt
from tokenizer import BOS_ID, EOS_ID, PAD_ID, decode, encode
from validate_data import ANSWERABILITY_VALUES, validate_data


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]  # Generated tokens only; includes EOS if predicted.
    ended_with_eos: bool
    valid_utf8: bool


def generate_answer(
    model: TinyGroundedQA,
    information: str,
    question: str,
    *,
    max_new_tokens: int | None = None,
    prompt_version: str = "original",
) -> GenerationResult:
    """Choose one token at a time, feeding back only the model's own output.

    There is no reference-answer argument. The budget includes EOS when the
    model predicts it. Input text is never truncated to fit the network.
    """
    limit = model.config.max_target_tokens if max_new_tokens is None else max_new_tokens
    if type(limit) is not int or not 1 <= limit <= model.config.max_target_tokens:
        raise ValueError(f"max_new_tokens must be between 1 and {model.config.max_target_tokens}.")
    source_tokens = encode(build_prompt(information, question, version=prompt_version), add_eos=True)
    if len(source_tokens) > model.config.max_input_tokens:
        raise ValueError(
            f"Prompt has {len(source_tokens)} tokens; input limit is "
            f"{model.config.max_input_tokens}. Nothing was truncated."
        )
    source_ids = torch.tensor([source_tokens], dtype=torch.long)
    generated = []
    ended_with_eos = False
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(limit):
                # First call: [BOS]. Later calls: [BOS] + our own predictions.
                # With a limit of 64, these input lengths range from 1 to 64.
                decoder_ids = torch.tensor([[BOS_ID] + generated], dtype=torch.long)
                logits = model(source_ids, decoder_ids)
                next_scores = logits[0, -1].clone()
                if not torch.isfinite(next_scores).all().item():
                    raise ValueError("Model produced non-finite next-token scores.")
                # PAD and BOS are input markers, never valid answer tokens.
                next_scores[PAD_ID] = -torch.inf
                next_scores[BOS_ID] = -torch.inf
                next_id = int(torch.argmax(next_scores).item())
                generated.append(next_id)
                if next_id == EOS_ID:
                    ended_with_eos = True
                    break
    finally:
        model.train(was_training)

    try:
        text = decode(generated)
        valid_utf8 = True
    except UnicodeDecodeError:
        # A small model may emit malformed bytes. Show them using replacement
        # characters, but do not count that display as a valid matching answer.
        text = decode(generated, errors="replace")
        valid_utf8 = False
    return GenerationResult(text, generated, ended_with_eos, valid_utf8)


def matches_reference(result: GenerationResult, reference: str) -> bool:
    """Require valid text, an actual end marker, and an exact text match."""
    return result.valid_utf8 and result.ended_with_eos and result.text == reference


def select_examples(
    examples: list[dict], example_id: str | None = None, source_id: str | None = None,
    source_ids: list[str] | None = None,
) -> list[dict]:
    """Select one question, complete source groups, or the first row of each type."""
    if sum(value is not None for value in (example_id, source_id, source_ids)) > 1:
        raise ValueError("Choose an example ID, a source ID, or a list of source IDs.")
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
            raise ValueError("Source IDs not found in the selected dataset: " + ", ".join(sorted(missing)))
        return [row for row in examples if row["source_id"] in requested]
    if source_id is not None:
        selected = [row for row in examples if row["source_id"] == source_id]
        if not selected:
            raise ValueError(f"Source ID not found in the selected dataset: {source_id}")
        return selected
    if example_id is not None:
        selected = [row for row in examples if row["example_id"] == example_id]
        if not selected:
            raise ValueError(f"Example ID not found: {example_id}")
        return selected
    # Labels choose the examples for OUR inspection; they are never model inputs.
    selected = []
    for label in ANSWERABILITY_VALUES:
        match = next((row for row in examples if row["answerability"] == label), None)
        if match is not None:
            selected.append(match)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="saved .pt model to use")
    parser.add_argument("--data", type=Path, help="reference dataset (default: data/training/generated_train.jsonl)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--example-id", help="preview one row instead of the first row of each type")
    selection.add_argument("--source-id", help="preview every question for one source, in file order")
    selection.add_argument(
        "--source-ids", nargs="+", metavar="ID",
        help="preview every question for these sources, in file order",
    )
    parser.add_argument("--information", help="custom supplied information; requires --question")
    parser.add_argument("--question", help="custom question; requires --information")
    parser.add_argument("--max-new-tokens", type=int, help="generation budget including EOS (default: model limit)")
    args = parser.parse_args()

    custom = args.information is not None or args.question is not None
    if custom and (args.information is None or args.question is None):
        parser.error("Supply both --information and --question.")
    if custom and any(value is not None for value in (
        args.data, args.example_id, args.source_id, args.source_ids,
    )):
        parser.error("Use either custom information/question or dataset options.")

    try:
        if custom:
            selected = [{"information": args.information, "question": args.question}]
        else:
            data_path = args.data if args.data is not None else Path("data/training/generated_train.jsonl")
            examples, errors = validate_data(data_path)
            if errors:
                raise ValueError("Dataset validation failed:\n  - " + "\n  - ".join(errors))
            selected = select_examples(examples, args.example_id, args.source_id, args.source_ids)

        torch.set_num_threads(1)
        model, training_info = load_checkpoint(args.checkpoint)
        prompt_version = training_info.get("prompt_version", "original")
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"Prompt version: {prompt_version}")
        print("Greedy generation on CPU: choose the highest-scoring allowed token each time.")
        print("The decoder receives BOS and its own predictions; no reference answer.")
        if not custom:
            print(f"Previewing {len(selected)} examples from {data_path}.")
            print(f"Source groups: {len({row['source_id'] for row in selected})}")
        match_count = 0
        for row in selected:
            if custom:
                print("\nCustom example")
            else:
                print(f"\nExample: {row['example_id']} ({row['answerability']})")
            # ascii() escapes control characters and supports Windows terminals.
            print(f"Information: {ascii(row['information'])}")
            print(f"Question: {ascii(row['question'])}", flush=True)
            result = generate_answer(
                model, row["information"], row["question"],
                max_new_tokens=args.max_new_tokens, prompt_version=prompt_version,
            )
            print(f"Generated answer: {ascii(result.text)}")
            stop = "EOS predicted" if result.ended_with_eos else "token limit reached without EOS"
            print(f"Stop: {stop}; generated tokens: {len(result.token_ids)}")
            print(f"Valid UTF-8: {result.valid_utf8}")
            if not custom:
                # Only AFTER generation do we consult the reference for comparison.
                matched = matches_reference(result, row["answer"])
                match_count += int(matched)
                print(f"Reference answer (comparison only): {ascii(row['answer'])}")
                print(f"Exact completed match: {matched}")

        if not custom:
            print(f"\nExact completed matches in this preview: {match_count}/{len(selected)}")
            print("This small preview does not establish performance on unseen sources.")
        print("No training was run. No dataset or checkpoint was changed.")
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        print(f"Prediction stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
