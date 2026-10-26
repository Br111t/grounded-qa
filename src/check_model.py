"""Check a random or reloaded CPU model on one example, without updating weights."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from model import TinyGroundedQA
from prompting import build_prompt
from tokenizer import PAD_ID, VOCAB_SIZE, encode
from validate_data import validate_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/training/generated_train.jsonl"))
    parser.add_argument("--example-id", help="example to inspect (default: first row)")
    parser.add_argument("--seed", type=int, default=42, help="random weight seed; unused with --checkpoint (default: 42)")
    parser.add_argument("--checkpoint", type=Path, help="load a saved .pt file instead of random weights")
    args = parser.parse_args()

    try:
        examples, errors = validate_data(args.data)
    except (OSError, UnicodeError) as error:
        print(f"Could not read {args.data}: {error}")
        return 1
    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    example = examples[0]
    if args.example_id is not None:
        matches = [row for row in examples if row["example_id"] == args.example_id]
        if not matches:
            print(f"Example ID not found in {args.data}: {args.example_id}")
            return 1
        example = matches[0]

    # Keep this small check on the CPU and avoid using all available threads.
    torch.set_num_threads(1)
    if args.checkpoint is None:
        torch.manual_seed(args.seed)
        model = TinyGroundedQA().to(device="cpu", dtype=torch.float32)
        weight_source = "random weights"
        prompt_version = "original"
    else:
        try:
            model, training_info = load_checkpoint(args.checkpoint)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Model check failed: {error}")
            return 1
        weight_source = "reloaded weights"
        prompt_version = training_info.get("prompt_version", "original")
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"Recorded training updates: {training_info['steps']}")
    model.eval()  # Disable dropout so the check is repeatable.

    prompt = build_prompt(example["information"], example["question"], version=prompt_version)
    source_ids = torch.tensor([encode(prompt, add_eos=True)], dtype=torch.long)
    # Teacher forcing: provide only the correct PRECEDING answer tokens.
    # For "Hi": decoder = [BOS, H, i]; labels = [H, i, EOS].
    decoder_ids = torch.tensor([encode(example["answer"], add_bos=True)], dtype=torch.long)
    labels = torch.tensor([encode(example["answer"], add_eos=True)], dtype=torch.long)

    try:
        with torch.no_grad():
            logits = model(source_ids, decoder_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=PAD_ID,
            )
    except ValueError as error:
        print(f"Model check failed: {error}")
        return 1
    if logits.shape != (*labels.shape, VOCAB_SIZE):
        print("Model check failed: logits do not match the target shape.")
        return 1
    if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
        print("Model check failed: logits or loss contain non-finite values.")
        return 1

    parameter_count = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"PyTorch: {torch.__version__}; device: CPU; weights: float32")
    print(f"Example: {example['example_id']} ({example['answerability']})")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Weight storage: {weight_bytes / 1024**2:.2f} MiB (weights only, not total runtime RAM)")
    print(f"Source shape: {tuple(source_ids.shape)} [batch, input tokens]")
    print(f"Decoder shape: {tuple(decoder_ids.shape)} [batch, answer tokens]")
    print(f"Logits shape: {tuple(logits.shape)} [batch, answer tokens, vocabulary]")
    print(f"Teacher-forced loss from {weight_source}: {loss.item():.4f}")
    print("Check passed. Finite loss checks the calculation, not answering accuracy.")
    print("This check updated no weights, saved no checkpoint, and changed no dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
