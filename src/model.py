"""A tiny encoder-decoder built locally with random weights and our tokenizer."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from tokenizer import BOS_ID, PAD_ID, VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    """Small starting dimensions for our CPU experiment."""

    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_input_tokens: int = 640
    max_target_tokens: int = 64

    def __post_init__(self) -> None:
        for name in (
            "d_model", "nhead", "num_encoder_layers", "num_decoder_layers",
            "dim_feedforward", "max_input_tokens", "max_target_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1.")


class TinyGroundedQA(nn.Module):
    """Read a prompt and score possible next tokens in an answer.

    Source: prompt tokens + EOS.
    Decoder input: BOS + preceding answer tokens.
    Output: raw scores (logits), shaped [batch, decoder length, vocabulary].

    This defines a network, not learned answering behavior. It loads no files.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        cfg = self.config

        # The same token has the same starting representation on both sides.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD_ID)
        self.source_positions = nn.Embedding(cfg.max_input_tokens, cfg.d_model)
        self.target_positions = nn.Embedding(cfg.max_target_tokens, cfg.d_model)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.output_head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    @staticmethod
    def _check_tokens(tokens: torch.Tensor, name: str, limit: int) -> torch.Tensor:
        """Validate a nonempty batch and return its padding mask."""
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError(f"{name} must have shape [batch, length] with neither dimension empty.")
        if tokens.dtype != torch.long:
            raise ValueError(f"{name} must contain torch.long token IDs.")
        if tokens.shape[1] > limit:
            raise ValueError(f"{name} length {tokens.shape[1]} exceeds limit {limit}; nothing was truncated.")
        if ((tokens < 0) | (tokens >= VOCAB_SIZE)).any().item():
            raise ValueError(f"{name} token IDs must be between 0 and {VOCAB_SIZE - 1}.")

        padding = tokens.eq(PAD_ID)
        if padding.all(dim=1).any().item():
            raise ValueError(f"{name} cannot contain an all-padding sequence.")
        # Once padding starts, every remaining position must also be padding.
        if (padding[:, :-1] & ~padding[:, 1:]).any().item():
            raise ValueError(f"{name} must use right padding only.")
        return padding

    def _embed(self, tokens: torch.Tensor, positions: nn.Embedding) -> torch.Tensor:
        position_ids = torch.arange(tokens.shape[1], device=tokens.device)
        # Token vectors carry identity; position vectors carry sequence order.
        return self.token_embedding(tokens) * math.sqrt(self.config.d_model) + positions(position_ids)

    def forward(self, source_ids: torch.Tensor, decoder_ids: torch.Tensor) -> torch.Tensor:
        source_padding = self._check_tokens(source_ids, "source_ids", self.config.max_input_tokens)
        target_padding = self._check_tokens(decoder_ids, "decoder_ids", self.config.max_target_tokens)
        if source_ids.shape[0] != decoder_ids.shape[0]:
            raise ValueError("Source and decoder inputs must have the same batch size.")
        if source_ids.device != decoder_ids.device:
            raise ValueError("Source and decoder inputs must be on the same device.")
        if not decoder_ids[:, 0].eq(BOS_ID).all().item():
            raise ValueError("Every decoder sequence must begin with BOS.")

        target_length = decoder_ids.shape[1]
        # For nn.Transformer, True blocks attention. A position can see itself
        # and earlier positions, but cannot peek at later answer tokens.
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, dtype=torch.bool, device=decoder_ids.device),
            diagonal=1,
        )
        # No padding mask is needed when every token is real.
        source_mask = source_padding if source_padding.any().item() else None
        target_mask = target_padding if target_padding.any().item() else None
        hidden = self.transformer(
            src=self._embed(source_ids, self.source_positions),
            tgt=self._embed(decoder_ids, self.target_positions),
            tgt_mask=causal_mask,
            src_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=source_mask,
        )
        # Cross-entropy expects these raw scores, so do not apply softmax here.
        return self.output_head(hidden)
