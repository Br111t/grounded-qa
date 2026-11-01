"""Checkpoint-free unit tests for V5D logit ranking and joint scoring."""
import unittest

from experiments.balanced_boundary_v5d.diagnose_logits import (
    boundary_rank, choose_cases, ranked_candidates, top_joint_spans,
)


class LogitDiagnosticTests(unittest.TestCase):
    def test_raw_ranks_only_permitted_evidence_positions(self):
        logits = [999, 1, 4, 3, 900]
        self.assertEqual(ranked_candidates(logits, [1, 2, 3], 2),
                         [{"position": 2, "score": 4.0},
                          {"position": 3, "score": 3.0}])
        self.assertEqual(boundary_rank(logits, [1, 2, 3], 1), 3)

    def test_joint_decoder_can_recover_wrong_raw_end(self):
        starts = [-100, 10, 0, 0]
        ends = [-100, 7, 5, 9]
        # Raw end 3; however max span length=1 makes (1, 1) best.
        self.assertEqual(ranked_candidates(ends, [1, 2, 3], 1)[0]["position"], 3)
        self.assertEqual(top_joint_spans(starts, ends, [1, 2, 3], 1, 1)[0],
                         {"start": 1, "end": 1, "score": 17.0})

    def test_joint_span_respects_max_length(self):
        starts = [0, 10, 1, 0]
        ends = [0, 0, 2, 100]
        spans = top_joint_spans(starts, ends, [1, 2, 3], 2, 10)
        self.assertFalse(any(s["start"] == 1 and s["end"] == 3 for s in spans))

    def test_deterministic_tie_break(self):
        starts = [0, 1, 1]
        ends = [0, 1, 1]
        self.assertEqual(top_joint_spans(starts, ends, [1, 2], 2, 1)[0],
                         {"start": 1, "end": 1, "score": 2.0})

    def test_representative_cases_are_deterministic(self):
        cases = [
            {"group": "length4_vowel", "error_category": "exact",
             "example_id": "b"},
            {"group": "length4_vowel", "error_category": "exact",
             "example_id": "a"},
        ]
        self.assertEqual([r["example_id"] for r in choose_cases(cases, 1)], ["a"])


if __name__ == "__main__":
    unittest.main()
