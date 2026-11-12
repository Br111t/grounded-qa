"""Invariants for V5G objective-head decomposition (no checkpoint required)."""
import unittest

from experiments.question_aware_v5g.diagnose_loss_components import aggregate, render


class V5GLossDiagnosticTests(unittest.TestCase):
    def test_aggregate_separates_answerable_and_unanswerable(self):
        a = {
            "answerable": True, "answerability_bce": 0.2,
            "answerability_correct": True, "start_ce": 0.4,
            "end_ce": 0.8, "extraction_ce": 0.6,
            "start_correct": True, "end_correct": False,
            "both_heads_correct": False,
        }
        b = {
            "answerable": False, "answerability_bce": 0.6,
            "answerability_correct": False, "start_ce": None,
            "end_ce": None, "extraction_ce": None,
            "start_correct": None, "end_correct": None,
            "both_heads_correct": None,
        }
        result = aggregate([a, b])
        self.assertEqual(result["answerable"], 1)
        self.assertEqual(result["unanswerable"], 1)
        self.assertAlmostEqual(result["answerability_bce_all"], 0.4)
        self.assertAlmostEqual(result["start_ce_answerable"], 0.4)
        self.assertAlmostEqual(result["end_ce_answerable"], 0.8)
        self.assertAlmostEqual(result["extraction_ce_answerable"], 0.6)
        self.assertAlmostEqual(result["mean_total_objective_all"], 0.7)
        self.assertAlmostEqual(result["both_heads_accuracy_answerable"], 0.0)

    def test_summary_reports_two_endpoints(self):
        row = {
            "rows": 2,
            "answerability_bce_all": 0.4,
            "answerability_bce_answerable": 0.2,
            "answerability_bce_unanswerable": 0.6,
            "start_ce_answerable": 0.4,
            "end_ce_answerable": 0.8,
            "extraction_ce_answerable": 0.6,
            "mean_total_objective_all": 0.7,
            "answerability_accuracy_all": 0.5,
            "start_head_accuracy_answerable": 1.0,
            "end_head_accuracy_answerable": 0.0,
            "both_heads_accuracy_answerable": 0.0,
        }
        data = {
            "results": {
                p: {"initial": row, "trained": row} for p in ("train", "dev")
            },
            "groups": {
                p: {"trained": {"field:leader": row}} for p in ("train", "dev")
            },
        }
        text = render(data)
        self.assertIn("TRAIN (2 questions)", text)
        self.assertIn("DEV (2 questions)", text)
        self.assertIn("start_ce_answerable", text)
        self.assertIn("end_ce_answerable", text)
        self.assertIn("answerability_bce_all", text)


if __name__ == "__main__":
    unittest.main()
