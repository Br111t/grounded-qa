"""V5: from-scratch encoder-only Transformer for extractive question answering.

The model predicts source-span start/end byte positions and whether the answer
exists. It never generates a name character from a vocabulary distribution.
"""

from contextlib import suppress
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import pickle

import torch
from torch import nn
from torch.nn import functional as F

from evidence_v3_validation import parse_fields
from prompting import build_prompt
from tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, PAD_ID, VOCAB_SIZE, decode, encode

FALLBACK = "Insufficient information."
MODEL_VARIANT = "span_v5"
SOURCE_PREFIX = "Information:\n"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SpanConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_input_tokens: int = 640
    max_span_bytes: int = 16

    def __post_init__(self):
        for name in (
            "d_model", "nhead", "num_layers", "dim_feedforward",
            "max_input_tokens", "max_span_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")


class SpanGroundedQA(nn.Module):
    """Predict start/end logits over the evidence and a no-answer decision."""

    def __init__(self, config: SpanConfig | None = None):
        super().__init__()
        self.config = config or SpanConfig()
        cfg = self.config
        self.embedding = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD_ID)
        self.positions = nn.Embedding(cfg.max_input_tokens, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.start_head = nn.Linear(cfg.d_model, 1)
        self.end_head = nn.Linear(cfg.d_model, 1)
        self.answerability_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self, token_ids: torch.Tensor, candidate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or not all(token_ids.shape):
            raise ValueError("Token IDs must be a nonempty [batch, length] tensor.")
        if token_ids.dtype != torch.long or token_ids.shape[1] > self.config.max_input_tokens:
            raise ValueError("Invalid token type or input length.")
        if candidate_mask.shape != token_ids.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("Candidate mask must be boolean with token shape.")
        if token_ids.device != candidate_mask.device:
            raise ValueError("Tokens and candidate mask must be on the same device.")
        if not token_ids[:, 0].eq(BOS_ID).all().item():
            raise ValueError("Every input must start with BOS.")
        if ((token_ids < 0) | (token_ids >= VOCAB_SIZE)).any().item():
            raise ValueError("Invalid token ID.")
        if not candidate_mask.any(dim=1).all().item():
            raise ValueError("Every row needs candidate evidence bytes.")
        if (~token_ids.ge(BYTE_OFFSET) & candidate_mask).any().item():
            raise ValueError("Candidates must be real text bytes.")
        padding = token_ids.eq(PAD_ID)
        if (padding[:, :-1] & ~padding[:, 1:]).any().item():
            raise ValueError("Only right padding is supported.")

        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        embedded = (
            self.embedding(token_ids) * math.sqrt(self.config.d_model)
            + self.positions(positions)
        )
        hidden = self.encoder(
            embedded,
            src_key_padding_mask=padding if padding.any().item() else None,
        )
        start = self.start_head(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        end = self.end_head(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        # BOS aggregates context from evidence and question via bidirectional encoder.
        answerability = self.answerability_head(hidden[:, 0]).squeeze(-1)
        return start, end, answerability


def prepare_span_input(
    information: str, question: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allow source spans only inside Information, never question or framing."""
    prompt = build_prompt(information, question, version="compact_v1")
    if not prompt.startswith(SOURCE_PREFIX + information + "\n\nQuestion:\n"):
        raise ValueError("Unexpected compact prompt structure.")
    token_ids = [BOS_ID] + encode(prompt, add_eos=True)
    info_start = 1 + len(encode(SOURCE_PREFIX))
    info_end = info_start + len(encode(information))
    mask = [False] * len(token_ids)
    for i in range(info_start, info_end):
        mask[i] = True
    if not information.strip():
        raise ValueError("Information must not be empty.")
    return (
        torch.tensor([token_ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.bool),
    )


def training_span_target(
    row: dict, token_ids: torch.Tensor, candidate_mask: torch.Tensor
) -> tuple[float, int | None, int | None]:
    """Derive gold indices from validated TRAINING reference, never at inference."""
    fields = parse_fields(row["information"], row["source_id"])
    leader = fields.get("Leader")
    if leader is None:
        if row["answerability"] != "unanswerable" or row["answer"] != FALLBACK:
            raise ValueError("Missing leader must have the correct fallback.")
        return 0.0, None, None
    if row["answerability"] != "answerable" or row["answer"] != f"Leader: {leader}.":
        raise ValueError("Leader reference does not agree with evidence.")
    marker = f"Leader: {leader}."
    if row["information"].count(marker) != 1:
        raise ValueError("Expected exactly one grounded leader field.")
    field_start = row["information"].index(marker) + len("Leader: ")
    prefix = SOURCE_PREFIX + row["information"][:field_start]
    start = 1 + len(encode(prefix))
    name_bytes = encode(leader)
    end = start + len(name_bytes) - 1
    if len(name_bytes) > SpanConfig().max_span_bytes:
        raise ValueError("Training name is longer than the model's span budget.")
    if not candidate_mask[0, start:end + 1].all().item():
        raise ValueError("Target span lies outside permitted evidence.")
    if token_ids[0, start:end + 1].tolist() != name_bytes:
        raise ValueError("Selected evidence bytes disagree with reference.")
    return 1.0, start, end


def span_training_loss(
    model: SpanGroundedQA, token_ids: torch.Tensor, mask: torch.Tensor,
    target: tuple[float, int | None, int | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_logits, end_logits, answer_logit = model(token_ids, mask)
    answerable, start, end = target
    classification = F.binary_cross_entropy_with_logits(
        answer_logit, torch.tensor([answerable], device=answer_logit.device)
    )
    if not answerable:
        zero = classification * 0.0
        return classification, classification, zero
    if start is None or end is None:
        raise ValueError("Answerable row lacks target span.")
    source_target = torch.tensor([start], device=start_logits.device)
    end_target = torch.tensor([end], device=end_logits.device)
    extraction = (
        F.cross_entropy(start_logits, source_target)
        + F.cross_entropy(end_logits, end_target)
    ) / 2
    return classification + extraction, classification, extraction


@torch.no_grad()
def predict_span(
    model: SpanGroundedQA, information: str, question: str,
    *, threshold: float = 0.5,
) -> tuple[str, float, tuple[int, int] | None]:
    if not 0 < threshold < 1:
        raise ValueError("Answerability threshold must be between 0 and 1.")
    tokens, mask = prepare_span_input(information, question)
    was_training = model.training
    model.eval()
    try:
        start_logits, end_logits, answer_logits = model(tokens, mask)
        probability = float(torch.sigmoid(answer_logits[0]).item())
        if probability < threshold:
            return FALLBACK, probability, None

        # Jointly choose the highest-scoring valid span, excluding framing and
        # question bytes. The chosen byte slice, not generated text, is returned.
        positions = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
        first, last = positions[0], positions[-1]
        best_score = -float("inf")
        best_span = None
        for start in range(first, last + 1):
            max_end = min(last, start + model.config.max_span_bytes - 1)
            end = start + int(torch.argmax(end_logits[0, start:max_end + 1]).item())
            score = float((start_logits[0, start] + end_logits[0, end]).item())
            if score > best_score:
                best_span = (start, end)
                best_score = score
        if best_span is None:
            raise ValueError("No valid source span could be selected.")
        start, end = best_span
        # Strict decode prevents malformed UTF-8 slices from appearing correct.
        try:
            value = decode(tokens[0, start:end + 1].tolist())
        except UnicodeDecodeError:
            value = "\ufffd"
        return f"Leader: {value}.", probability, best_span
    finally:
        model.train(was_training)


def save_span_checkpoint(
    path: Path, model: SpanGroundedQA, training_info: dict
) -> None:
    if path.suffix.lower() != ".pt" or path.exists() or path.is_symlink():
        raise ValueError("Use a new .pt checkpoint filename; overwriting is forbidden.")
    training_info = json.loads(json.dumps(training_info, allow_nan=False))
    payload = {
        "format_version": FORMAT_VERSION,
        "model_variant": MODEL_VARIANT,
        "model_config": asdict(model.config),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "training": training_info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            torch.save(payload, handle)
    except BaseException:
        if created:
            with suppress(OSError):
                path.unlink()
        raise


def load_span_checkpoint(path: Path) -> tuple[SpanGroundedQA, dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError(f"Cannot load V5 checkpoint: {error}") from error
    if (not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION
            or payload.get("model_variant") != MODEL_VARIANT):
        raise ValueError("Not a supported V5 span checkpoint.")
    data = payload.get("model_config")
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(SpanConfig)}:
        raise ValueError("Invalid V5 model configuration.")
    model = SpanGroundedQA(SpanConfig(**data))
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing checkpoint weights.")
    if any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor)
        or v.dtype != torch.float32 or not torch.isfinite(v).all().item()
        for k, v in state.items()
    ):
        raise ValueError("Invalid checkpoint weight tensors.")
    model.load_state_dict(state, strict=True)
    info = payload.get("training")
    if not isinstance(info, dict) or not isinstance(info.get("steps"), int):
        raise ValueError("V5 checkpoint is missing training metadata.")
    model.eval()
    return model, info
