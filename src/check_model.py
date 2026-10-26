"""Check a random or reloaded CPU model on one example, without updating weights."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from copy_model import CopyGroundedQA
from model import TinyGroundedQA
from prompting import build_prompt
from tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, PAD_ID, VOCAB_SIZE, decode, encode
from validate_data import validate_data



def trace_copy_details(
    model: CopyGroundedQA, information: str, question: str,
    reference: str, *, prompt_version: str,
) -> str:
    """Inspect ACTUAL greedy name characters, using the model's generated prefix.

    The reference is used only to annotate output, never for decoder input.
    """
    prompt = build_prompt(information, question, version=prompt_version)
    source = torch.tensor([encode(prompt, add_eos=True)], dtype=torch.long)
    reference_bytes = encode(reference)
    generated: list[int] = []
    displayed = 0

    # Source positions correspond to UTF-8 bytes of the complete prompt.
    # Mark the value region of the provided Information, not the Question.
    marker = "Leader: "
    name_span = None
    field_start = information.find(marker)
    if field_start >= 0:
        name_start = field_start + len(marker)
        name_end = information.find(".", name_start)
        if name_end > name_start:
            start_in_prompt = prompt.find(information)
            if start_in_prompt >= 0:
                char_start = start_in_prompt + name_start
                char_end = start_in_prompt + name_end
                name_span = (
                    len(prompt[:char_start].encode("utf-8")),
                    len(prompt[:char_end].encode("utf-8")),
                )
    print("\nCOPY DETAILS — free-running greedy output (not teacher-forced)")
    print("Generate fraction = gate for vocabulary route; copy = 1 - generate.")
    print("Probability contributions are gate * vocabulary and (1-gate) * copy.")
    if name_span is None:
        print("No Leader field detected in supplied information.")
    else:
        print(f"Supplied leader bytes: source positions [{name_span[0]}, {name_span[1]})")

    with torch.no_grad():
        for _ in range(model.config.max_target_tokens):
            decoder = torch.tensor([[BOS_ID] + generated], dtype=torch.long)
            log_prob, parts = model.inspect_copy(source, decoder)
            scores = log_prob[0, -1].clone()
            scores[PAD_ID] = float("-inf")
            scores[BOS_ID] = float("-inf")
            token = int(torch.argmax(scores).item())

            # Only display alphabetic bytes in an actually generated "Leader: <name>"
            # region; do not confuse it with teacher-forced reference characters.
            prefix = decode(generated, errors="replace")
            letter_byte = token - BYTE_OFFSET
            is_letter = (
                65 <= letter_byte <= 90 or 97 <= letter_byte <= 122
            )
            in_name = (
                prefix.startswith("Leader: ")
                and "." not in prefix[len("Leader: "):]
                and is_letter
            )
            if in_name:
                displayed += 1
                position = len(generated)
                generate = float(parts["generate_fraction"][0, -1, 0].item())
                vocabulary = float(parts["vocabulary_probability"][0, -1, token].item())
                copy = float(parts["copy_probability"][0, -1, token].item())
                combined = float(parts["combined_probability"][0, -1, token].item())
                attention = parts["copy_attention"][0, -1]
                attention_on_leader = (
                    float(attention[name_span[0]:name_span[1]].sum().item())
                    if name_span is not None else 0.0
                )
                attention_on_matching_leader_byte = 0.0
                if name_span is not None:
                    leader_attention = attention[name_span[0]:name_span[1]]
                    leader_ids = source[0, name_span[0]:name_span[1]]
                    attention_on_matching_leader_byte = float(
                        leader_attention[leader_ids.eq(token)].sum().item()
                    )
                top = torch.topk(attention, k=min(3, attention.numel())).indices.tolist()
                source_hits = []
                for index in top:
                    byte_id = int(source[0, index].item()) - BYTE_OFFSET
                    character = chr(byte_id) if 0 <= byte_id < 128 else f"byte{byte_id}"
                    region = " LEADER_VALUE" if (
                        name_span is not None and name_span[0] <= index < name_span[1]
                    ) else ""
                    source_hits.append(
                        f"{index}:{character!r} {float(attention[index]):.3f}{region}"
                    )
                aligned = generated == reference_bytes[:len(generated)]
                expected_note = ""
                if aligned and position < len(reference_bytes):
                    expected_token = reference_bytes[position]
                    expected_char = (
                        chr(expected_token - BYTE_OFFSET)
                        if expected_token >= BYTE_OFFSET else "<special>"
                    )
                    p_expected = float(parts["combined_probability"][0, -1, expected_token].item())
                    expected_note = f"; ref next={expected_char!r} P(ref)={p_expected:.3f}"
                print(
                    f"  output byte {position + 1:>2}: {chr(token - BYTE_OFFSET)!r}; "
                    f"generate={generate:.3f}, copy={1-generate:.3f}; "
                    f"P_vocab={vocabulary:.3f}, P_copy={copy:.3f}, "
                    f"P_final={combined:.3f}{expected_note}"
                )
                print(
                    f"    weighted gen={generate * vocabulary:.3f}, "
                    f"weighted copy={(1-generate)*copy:.3f}; "
                    f"attention on leader value={attention_on_leader:.3f}, "
                    f"attention on matching leader bytes={attention_on_matching_leader_byte:.3f}"
                )
                print("    top source positions: " + "; ".join(source_hits))
            generated.append(token)
            if token == EOS_ID:
                break

    generated_text = decode(generated, errors="replace")
    print(f"Greedy generated answer: {generated_text!r}")
    print(f"Reference (comparison only): {reference!r}")
    if not displayed:
        print("No generated leader-name characters to inspect.")
    print("Attention values are from the learned copy pointer, not Transformer cross-attention.")
    print("The diagnostic did not alter weights or checkpoint.")
    return generated_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/training/generated_train.jsonl"))
    parser.add_argument("--example-id", help="example to inspect (default: first row)")
    parser.add_argument("--seed", type=int, default=42, help="random weight seed; unused with --checkpoint (default: 42)")
    parser.add_argument("--checkpoint", type=Path, help="load a saved .pt file instead of random weights")
    parser.add_argument(
        "--copy-details", action="store_true",
        help="trace actual generated name bytes and V4 copy/generate contributions",
    )
    args = parser.parse_args()
    if args.copy_details and args.checkpoint is None:
        parser.error("--copy-details requires a saved V4 checkpoint.")

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
    if args.copy_details and not isinstance(model, CopyGroundedQA):
        print("Model check failed: --copy-details is only available for V4 pointer checkpoints.")
        return 1

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
    if args.copy_details:
        trace_copy_details(
            model, example["information"], example["question"], example["answer"],
            prompt_version=prompt_version,
        )
    print("This check updated no weights, saved no checkpoint, and changed no dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
