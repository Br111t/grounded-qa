"""Regression tests for the first cooperative-agent contract."""
import unittest

from grounded_qa.agent_contracts import (
    AgentReview,
    AnswerProposal,
    EvidenceSpan,
    verify_literal_grounding,
)


class AgentContractTests(unittest.TestCase):
    def setUp(self):
        self.evidence = "Originally Marina led. After transition, Elena led."

    def test_exact_evidence_span_is_accepted(self):
        start = self.evidence.index("Elena")
        proposal = AnswerProposal("extractor", "Elena", EvidenceSpan(start, start + 5))
        review = verify_literal_grounding(proposal, self.evidence)
        self.assertTrue(review.accepted)
        self.assertEqual(review.reason, "literal_span_match")

    def test_incorrect_answer_for_span_is_rejected(self):
        start = self.evidence.index("Elena")
        proposal = AnswerProposal("extractor", "Marina", EvidenceSpan(start, start + 5))
        review = verify_literal_grounding(proposal, self.evidence)
        self.assertFalse(review.accepted)
        self.assertEqual(review.requested_action, "reconsider")

    def test_missing_span_is_rejected(self):
        proposal = AnswerProposal("extractor", "Elena", None)
        self.assertFalse(verify_literal_grounding(proposal, self.evidence).accepted)

    def test_out_of_range_span_is_rejected(self):
        proposal = AnswerProposal("extractor", "Elena", EvidenceSpan(400, 405))
        self.assertFalse(verify_literal_grounding(proposal, self.evidence).accepted)

    def test_empty_span_is_rejected(self):
        proposal = AnswerProposal("extractor", "", EvidenceSpan(3, 3))
        self.assertFalse(verify_literal_grounding(proposal, self.evidence).accepted)

    def test_literal_match_does_not_imply_correct_answer(self):
        # Marina occurs in the source, but is not the post-transition leader.
        start = self.evidence.index("Marina")
        proposal = AnswerProposal("extractor", "Marina", EvidenceSpan(start, start + 6))
        self.assertTrue(verify_literal_grounding(proposal, self.evidence).accepted)
        # Semantic/temporal support must be checked by a future verifier.


if __name__ == "__main__":
    unittest.main()
