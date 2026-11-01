"""Shared, model-independent messages for cooperative QA components.

Spans use Python half-open character offsets [start, end). Adapters must
convert any model-specific token/byte offsets before constructing these.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EvidenceSpan:
    start: int
    end: int

    def extract(self, evidence: str) -> str:
        if not (0 <= self.start < self.end <= len(evidence)):
            raise ValueError("Evidence span is outside the supplied text")
        return evidence[self.start:self.end]


@dataclass(frozen=True)
class AnswerProposal:
    agent_id: str
    answer: str
    span: Optional[EvidenceSpan]
    round_number: int = 0

    def is_grounded(self, evidence: str) -> bool:
        """Check literal provenance, not semantic correctness."""
        if self.span is None:
            return False
        try:
            return self.span.extract(evidence) == self.answer
        except ValueError:
            return False


@dataclass(frozen=True)
class AgentReview:
    reviewer_id: str
    accepted: bool
    reason: str
    requested_action: str = "none"


def verify_literal_grounding(proposal: AnswerProposal, evidence: str) -> AgentReview:
    """Minimal deterministic verifier; semantic checking is future work."""
    valid = proposal.is_grounded(evidence)
    return AgentReview(
        reviewer_id="literal_grounding_verifier",
        accepted=valid,
        reason="literal_span_match" if valid else "missing_or_mismatched_span",
        requested_action="none" if valid else "reconsider",
    )
