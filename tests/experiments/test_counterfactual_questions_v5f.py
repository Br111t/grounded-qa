"""V5F counterfactual experiment invariants (no checkpoints needed)."""
import unittest

import torch

from experiments.counterfactual_questions_v5f.probe_questions import (
    EVIDENCE, QUESTIONS, compare_pairs, design_rows, probe_one, summarize,
)
from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans


class FakeConfig:
    max_span_bytes = 16


class FakeModel(torch.nn.Module):
    """Question-sensitive synthetic logits for testing scoring, not training."""
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()

    def forward(self, tokens, mask):
        # An intentionally deterministic constant score on source positions.
        shape = tokens.shape
        starts = torch.zeros(shape)
        ends = torch.zeros(shape)
        return starts, ends, torch.tensor([8.0])


class CounterfactualQuestionTests(unittest.TestCase):
    def test_only_question_changes_within_evidence(self):
        rows = design_rows()
        self.assertEqual(len(rows), len(EVIDENCE) * len(QUESTIONS))
        for evidence_id, _ in EVIDENCE:
            subset = [r for r in rows if r["evidence_id"] == evidence_id]
            self.assertEqual(len(subset), len(QUESTIONS))
            self.assertEqual(len({r["information"] for r in subset}), 1)
            self.assertEqual(len({r["information_sha256"] for r in subset}), 1)
            self.assertEqual(len({r["question"] for r in subset}), len(QUESTIONS))
            self.assertEqual(
                {r["target_field"] for r in subset},
                {"leader", "location", "topic", "unrelated"},
            )

    def test_paraphrases_have_same_reference_field(self):
        rows = [r for r in design_rows() if r["evidence_id"] == "a"]
        leaders = [r for r in rows if r["target_field"] == "leader"]
        self.assertEqual(len(leaders), 3)
        self.assertTrue(all(r["expected_value"] == "Dbfi" for r in leaders))

    def test_candidate_values_are_source_derived(self):
        for _, information in EVIDENCE:
            fields = complete_field_spans(information)
            self.assertEqual({r["field"] for r in fields},
                             {"Leader", "Location", "Topic"})
            self.assertTrue(all(r["value"] in information for r in fields))

    def test_fixed_logits_yield_same_field_for_all_questions(self):
        rows = [r for r in design_rows() if r["evidence_id"] == "a"]
        model = FakeModel()
        predictions = [probe_one(model, r) for r in rows]
        self.assertEqual(len({p["predicted_field"] for p in predictions}), 1)
        self.assertEqual(len({tuple(sorted(p["scores_by_field"].items()))
                              for p in predictions}), 1)
        self.assertFalse(predictions[-1]["unrelated_abstained"])

    def test_pair_deltas_are_zero_with_constant_logits(self):
        model = FakeModel()
        records = [
            {"model": "v5", **r, **probe_one(model, r)}
            for r in design_rows()
        ]
        pairs = compare_pairs(records)
        self.assertEqual(len(pairs), len(EVIDENCE) * (len(QUESTIONS) - 1))
        self.assertTrue(all(not r["selection_changed"] for r in pairs))
        self.assertTrue(all(all(v == 0.0 for v in r["score_deltas"].values())
                            for r in pairs))

    def test_unrelated_question_is_marked_as_abstention_control(self):
        rows = [r for r in design_rows() if r["target_field"] == "unrelated"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["expected_field"] is None
                            and r["expect_abstain"] for r in rows))


if __name__ == "__main__":
    unittest.main()
