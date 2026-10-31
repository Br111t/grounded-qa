"""V4B supervision targets derived ONLY from validated training examples.

The teacher supplies source-position alignment during training only; checkpoint
inference remains exactly the V4 pointer generator with no position labels.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields
from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.tokenization.tokenizer import encode


@dataclass(frozen=True)
class NameAlignment:
    """Matching decoder-output and source positions for every name byte."""

    output_positions: tuple[int, ...]
    source_positions: tuple[int, ...]


def derive_name_alignment(row: dict, *, prompt_version: str) -> NameAlignment | None:
    """Fail closed if an answerable training row does not contain a matching name."""
    if prompt_version != "compact_v1":
        raise ValueError("V4B supervision requires compact_v1.")
    fields = parse_fields(row["information"], row["source_id"])
    leader = fields.get("Leader")
    if leader is None:
        if row["answerability"] != "unanswerable" or row["answer"] != "Insufficient information.":
            raise ValueError("Missing leader must use the V3 fallback reference.")
        return None
    if row["answerability"] != "answerable" or row["answer"] != f"Leader: {leader}.":
        raise ValueError("Answer does not match evidence leader; alignment refused.")

    marker = f"Leader: {leader}."
    information = row["information"]
    if information.count(marker) != 1:
        raise ValueError("Expected exactly one leader field in training information.")
    prompt = build_prompt(information, row["question"], version=prompt_version)
    if prompt.count(information) != 1:
        raise ValueError("Information must occur exactly once within the prompt.")
    source_char_start = prompt.index(information) + information.index(marker) + len("Leader: ")
    output_char_start = len("Leader: ")
    # Encode prefixes rather than treating Unicode character offsets as byte offsets.
    source_start = len(encode(prompt[:source_char_start]))
    target_start = len(encode(row["answer"][:output_char_start]))
    name_bytes = encode(leader)
    source_tokens = encode(prompt, add_eos=True)
    target_tokens = encode(row["answer"], add_eos=True)
    outputs = tuple(target_start + i for i in range(len(name_bytes)))
    sources = tuple(source_start + i for i in range(len(name_bytes)))
    if not outputs or any(
        source_tokens[src_pos] != target_tokens[out_pos]
        for out_pos, src_pos in zip(outputs, sources)
    ):
        raise ValueError("Source/answer name bytes are not aligned.")
    return NameAlignment(outputs, sources)


def auxiliary_copy_losses(
    parts: dict[str, torch.Tensor],
    alignment: NameAlignment | None,
    *,
    alignment_weight: float,
    gate_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two unweighted scalar auxiliary losses.

    Auxiliary losses are calculated only on answerable leader-NAME bytes.
    Unanswerable rows use only ordinary answer cross-entropy. For one batch
    of size 1, the supervision targets are positions in the output and prompt.
    """
    attention = parts["copy_attention"]
    gate = parts["generate_fraction"]
    if alignment is None or (alignment_weight == 0 and gate_weight == 0):
        zero = gate.sum() * 0.0
        return zero, zero
    if attention.ndim != 3 or gate.ndim != 3 or attention.shape[0] != 1 or gate.shape[:2] != attention.shape[:2] or gate.shape[-1] != 1:
        raise ValueError("Unexpected copy components; expected batch size one.")
    targets = torch.tensor(alignment.output_positions, dtype=torch.long, device=attention.device)
    positions = torch.tensor(alignment.source_positions, dtype=torch.long, device=attention.device)
    if targets.max().item() >= attention.shape[1] or positions.max().item() >= attention.shape[2]:
        raise ValueError("Supervised alignment exceeds model sequence lengths.")

    alignment_loss = gate.sum() * 0.0
    gate_loss = gate.sum() * 0.0
    if alignment_weight > 0:
        selected_attention = attention[0, targets, positions]
        alignment_loss = -torch.log(selected_attention.clamp_min(1e-12)).mean()
    if gate_weight > 0:
        # Gate is P(generate); encourage P(copy) close to 1 on name bytes.
        copy_fraction = 1.0 - gate[0, targets, 0]
        gate_loss = -torch.log(copy_fraction.clamp_min(1e-12)).mean()
    return alignment_loss, gate_loss
