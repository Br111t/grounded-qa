"""V5I: explicit question-conditioned readout over the unchanged V5 encoder.

The extra module learns an interaction between a pooled question vector and
each contextual token. V5H's data, ranking objective, and decoder are unchanged.
No label or requested field is passed to this module at inference.
"""
from dataclasses import asdict, fields
import json
import math
from pathlib import Path

import torch
from torch import nn

from grounded_qa.models.span_model_v5 import (
    SpanConfig, SpanGroundedQA, FORMAT_VERSION,
)
from grounded_qa.tokenization.tokenizer import BYTE_OFFSET, BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE, encode

MODEL_VARIANT = "span_v5i_question_conditioned"
QUESTION_SEPARATOR = encode("\n\nQuestion:\n")
ANSWER_SUFFIX = encode("\n\nAnswer:\n")


class QuestionConditionedSpanQA(SpanGroundedQA):
    """Add one trainable d_model x (2*d_model) fusion before the three heads."""

    def __init__(self, config: SpanConfig | None = None):
        super().__init__(config)
        self.question_fusion = nn.Linear(2 * self.config.d_model, self.config.d_model)

    @staticmethod
    def question_mask(token_ids: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """Find question bytes after the evidence, checking the actual separator."""
        mask = torch.zeros_like(candidate_mask)
        for row in range(token_ids.shape[0]):
            evidence = torch.nonzero(candidate_mask[row], as_tuple=False).flatten()
            if evidence.numel() == 0:
                raise ValueError("Evidence bytes missing")
            end = int(evidence[-1].item())
            sep_start = end + 1
            sep_end = sep_start + len(QUESTION_SEPARATOR)
            if token_ids[row, sep_start:sep_end].tolist() != QUESTION_SEPARATOR:
                raise ValueError("Expected compact question separator after evidence")
            remaining = token_ids[row, sep_end:].tolist()
            suffix_length = len(ANSWER_SUFFIX)
            suffix_at = next(
                (i for i in range(len(remaining) - suffix_length)
                 if remaining[i:i + suffix_length] == ANSWER_SUFFIX),
                None,
            )
            if suffix_at is None or suffix_at == 0:
                raise ValueError("Question missing or Answer suffix absent")
            suffix_end = suffix_at + suffix_length
            if (suffix_end >= len(remaining) or remaining[suffix_end] != EOS_ID
                    or any(value != PAD_ID for value in remaining[suffix_end + 1:])):
                raise ValueError("Invalid compact prompt termination")
            if any(value < BYTE_OFFSET for value in remaining[:suffix_at]):
                raise ValueError("Question contains a non-byte token")
            mask[row, sep_end:sep_end + suffix_at] = True
        return mask

    def forward(self, token_ids: torch.Tensor, candidate_mask: torch.Tensor):
        # Preserve V5's input constraints; do not invoke its heads.
        if token_ids.ndim != 2 or not all(token_ids.shape):
            raise ValueError("Token IDs must be nonempty [batch,length]")
        if token_ids.dtype != torch.long or token_ids.shape[1] > self.config.max_input_tokens:
            raise ValueError("Invalid token dtype or length")
        if candidate_mask.shape != token_ids.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("Invalid evidence candidate mask")
        if token_ids.device != candidate_mask.device:
            raise ValueError("Token/mask device mismatch")
        if not token_ids[:, 0].eq(BOS_ID).all().item():
            raise ValueError("BOS expected")
        if ((token_ids < 0) | (token_ids >= VOCAB_SIZE)).any().item():
            raise ValueError("Invalid token ID")
        if (~token_ids.ge(BYTE_OFFSET) & candidate_mask).any().item():
            raise ValueError("Candidate mask includes nonbyte token")
        padding = token_ids.eq(PAD_ID)
        if (padding[:, :-1] & ~padding[:, 1:]).any().item():
            raise ValueError("Only right padding supported")
        qmask = self.question_mask(token_ids, candidate_mask)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        embedded = self.embedding(token_ids) * math.sqrt(self.config.d_model) + self.positions(positions)
        hidden = self.encoder(
            embedded, src_key_padding_mask=padding if padding.any().item() else None
        )
        question = (hidden * qmask.unsqueeze(-1)).sum(dim=1) / qmask.sum(dim=1, keepdim=True)
        combined = torch.cat((hidden, question.unsqueeze(1).expand_as(hidden)), dim=-1)
        conditioned = hidden + torch.tanh(self.question_fusion(combined))
        start = self.start_head(conditioned).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        end = self.end_head(conditioned).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        answerability = self.answerability_head(conditioned[:, 0]).squeeze(-1)
        return start, end, answerability


def save_v5i_checkpoint(path: Path, model: QuestionConditionedSpanQA, training_info: dict) -> None:
    if not isinstance(model, QuestionConditionedSpanQA):
        raise TypeError("Expected question-conditioned model")
    if path.suffix.lower() != ".pt" or path.exists() or path.is_symlink():
        raise ValueError("Checkpoint must use a new .pt path")
    metadata = json.loads(json.dumps(training_info, allow_nan=False))
    payload = {
        "format_version": FORMAT_VERSION,
        "model_variant": MODEL_VARIANT,
        "model_config": asdict(model.config),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "training": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            torch.save(payload, stream)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_v5i_checkpoint(path: Path) -> tuple[QuestionConditionedSpanQA, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict)
            or payload.get("model_variant") != MODEL_VARIANT
            or payload.get("format_version") != FORMAT_VERSION):
        raise ValueError("Not a V5I checkpoint")
    config = payload.get("model_config")
    if not isinstance(config, dict) or set(config) != {f.name for f in fields(SpanConfig)}:
        raise ValueError("Invalid V5I config")
    model = QuestionConditionedSpanQA(SpanConfig(**config))
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor)
        or v.dtype != torch.float32 or not torch.isfinite(v).all().item()
        for k, v in state.items()
    ):
        raise ValueError("Invalid V5I model tensors")
    model.load_state_dict(state, strict=True)
    info = payload.get("training")
    if not isinstance(info, dict) or not isinstance(info.get("steps"), int):
        raise ValueError("Invalid V5I training metadata")
    model.eval()
    return model, info
