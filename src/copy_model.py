"""V4 pointer-generator extension of the original tiny Transformer.

Uses the original encoder, decoder, and vocabulary output head unchanged.
A learned source-token pointer and generation gate add a second path to the
next-token distribution. Trained from random weights; no pretrained language model.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model import ModelConfig, TinyGroundedQA
from tokenizer import BOS_ID, BYTE_OFFSET, PAD_ID, VOCAB_SIZE


class CopyGroundedQA(TinyGroundedQA):
    """Return normalized log probabilities [batch, target length, vocabulary].

    The pointer distribution sums attention for every occurrence of the same
    byte in the source. EOS is generated, not copied. This does not guarantee
    correct grounding; the training experiment measures whether it helps.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config)
        d_model = self.config.d_model
        self.copy_query = nn.Linear(d_model, d_model, bias=False)
        self.copy_key = nn.Linear(d_model, d_model, bias=False)
        self.generate_gate = nn.Linear(d_model * 2, 1)
        # Begin with both routes available; don't force copying or generation.
        nn.init.zeros_(self.generate_gate.weight)
        nn.init.zeros_(self.generate_gate.bias)

    def _forward_components(
        self, source_ids: torch.Tensor, decoder_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the exact V4 distribution and expose its constituent paths."""
        source_padding = self._check_tokens(
            source_ids, "source_ids", self.config.max_input_tokens
        )
        target_padding = self._check_tokens(
            decoder_ids, "decoder_ids", self.config.max_target_tokens
        )
        if source_ids.shape[0] != decoder_ids.shape[0]:
            raise ValueError("Source and decoder inputs must have the same batch size.")
        if source_ids.device != decoder_ids.device:
            raise ValueError("Source and decoder inputs must be on the same device.")
        if not decoder_ids[:, 0].eq(BOS_ID).all().item():
            raise ValueError("Every decoder sequence must begin with BOS.")

        target_length = decoder_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                target_length, target_length,
                dtype=torch.bool, device=decoder_ids.device,
            ),
            diagonal=1,
        )
        source_mask = source_padding if source_padding.any().item() else None
        target_mask = target_padding if target_padding.any().item() else None

        source_embeddings = self._embed(source_ids, self.source_positions)
        target_embeddings = self._embed(decoder_ids, self.target_positions)
        memory = self.transformer.encoder(
            source_embeddings, src_key_padding_mask=source_mask,
        )
        hidden = self.transformer.decoder(
            target_embeddings, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=source_mask,
        )

        vocabulary_prob = F.softmax(self.output_head(hidden), dim=-1)
        pointer_scores = torch.matmul(
            self.copy_query(hidden),
            self.copy_key(memory).transpose(1, 2),
        ) / math.sqrt(self.config.d_model)

        # PAD and EOS cannot be sources of copied text; generated EOS remains
        # available through the vocabulary route. Mask before softmax.
        copyable = source_ids.ge(BYTE_OFFSET)
        if not copyable.any(dim=1).all().item():
            raise ValueError("Every source sequence must contain copyable text.")
        pointer_scores = pointer_scores.masked_fill(
            ~copyable.unsqueeze(1), float("-inf")
        )
        pointer_attention = F.softmax(pointer_scores, dim=-1)

        copy_prob = torch.zeros(
            (*hidden.shape[:2], VOCAB_SIZE),
            dtype=vocabulary_prob.dtype, device=hidden.device,
        )
        source_index = source_ids.unsqueeze(1).expand(
            -1, target_length, -1
        )
        copy_prob = copy_prob.scatter_add(2, source_index, pointer_attention)
        context = torch.matmul(pointer_attention, memory)
        generate_fraction = torch.sigmoid(
            self.generate_gate(torch.cat((hidden, context), dim=-1))
        )
        probability = (
            generate_fraction * vocabulary_prob
            + (1.0 - generate_fraction) * copy_prob
        )
        # Use identical calculations for training/inference and the diagnostic.
        log_probability = probability.clamp_min(1e-12).log()
        components = {
            "generate_fraction": generate_fraction,
            "vocabulary_probability": vocabulary_prob,
            "copy_probability": copy_prob,
            "copy_attention": pointer_attention,
            "combined_probability": probability,
        }
        return log_probability, components

    def forward(self, source_ids: torch.Tensor, decoder_ids: torch.Tensor) -> torch.Tensor:
        """Return log probabilities, preserving V4 training and greedy prediction."""
        log_probability, _ = self._forward_components(source_ids, decoder_ids)
        return log_probability

    def inspect_copy(
        self, source_ids: torch.Tensor, decoder_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read-only copy-path diagnostics; caller must use eval() and no_grad()."""
        if self.training:
            raise ValueError("Copy diagnostics require model.eval() (dropout disabled).")
        return self._forward_components(source_ids, decoder_ids)
