"""Inference-only complete-field boundary decoder.

Candidates are the VALUE portions of every syntactically complete evidence
field, never reference answers or gold offsets. This is a deliberately strong
syntactic constraint, NOT a claim of general question understanding.
"""
import re

import torch

from grounded_qa.models.span_model_v5 import (
    FALLBACK, SOURCE_PREFIX, prepare_span_input,
)
from grounded_qa.tokenization.tokenizer import decode, encode

# Restrict to the evidence grammar without hard-coding the target field name.
FIELD_PATTERN = re.compile(r"(?:^|\.\s+)([A-Za-z]+): ([^\r\n.]+)\.(?=\s|$)")


def complete_field_spans(information, max_span_bytes=16):
    """Enumerate all source field values, using inclusive model-byte positions."""
    candidates = []
    for match in FIELD_PATTERN.finditer(information):
        label = match.group(1)
        value = match.group(2)
        size = len(encode(value))
        if not 1 <= size <= max_span_bytes:
            continue
        start = 1 + len(encode(SOURCE_PREFIX + information[:match.start(2)]))
        candidates.append({
            "field": label, "value": value,
            "start": start, "end": start + size - 1,
        })
    return candidates


@torch.no_grad()
def predict_complete_field(model, information, question, threshold=0.5):
    """Choose highest joint-logit complete field; leave answerability unchanged."""
    if not 0 < threshold < 1:
        raise ValueError("threshold outside (0,1)")
    tokens, mask = prepare_span_input(information, question)
    was_training = model.training
    model.eval()
    try:
        starts, ends, answer_logits = model(tokens, mask)
        probability = float(torch.sigmoid(answer_logits[0]).item())
        if probability < threshold:
            return FALLBACK, probability, None, []
        candidates = complete_field_spans(information, model.config.max_span_bytes)
        scored = []
        for c in candidates:
            s, e = c["start"], c["end"]
            if not mask[0, s:e + 1].all().item():
                raise ValueError("Field candidate escaped evidence mask")
            if decode(tokens[0, s:e + 1].tolist()) != c["value"]:
                raise ValueError("Field offset encoding mismatch")
            scored.append({
                **c, "score": float((starts[0, s] + ends[0, e]).item()),
            })
        scored.sort(key=lambda c: (-c["score"], c["start"], c["end"]))
        if not scored:
            # Do not fabricate a complete field if syntax is unsupported.
            return FALLBACK, probability, None, scored
        winner = scored[0]
        return f"Leader: {winner['value']}.", probability, (winner["start"], winner["end"]), scored
    finally:
        model.train(was_training)
