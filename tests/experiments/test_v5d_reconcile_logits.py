"""Tests for independent correctness and recovery bookkeeping."""
import unittest

from experiments.balanced_boundary_v5d.reconcile_logits import enrich, measure


def case(*, raw_end_correct, selected_span_correct, best_end=2.0,
         gold_end=1.0, best_joint=4.0, gold_start=1.0, exact=None):
    return {
        "raw_end_correct": raw_end_correct,
        "selected_span_correct": selected_span_correct,
        "exact": selected_span_correct if exact is None else exact,
        "selected_start": 6,
        "best_end_score": best_end,
        "gold_end_score": gold_end,
        "gold_start_score": gold_start,
        "top_joint_spans": [{"score": best_joint}],
        "gold_end_rank": 2,
    }


class ReconcileTests(unittest.TestCase):
    def test_recovery_is_also_correct(self):
        r = enrich(case(raw_end_correct=False, selected_span_correct=True))
        self.assertTrue(r["exact"])
        self.assertTrue(r["recovered_wrong_raw_end"])
        self.assertFalse(r["lost_correct_raw_end"])

    def test_wrong_span_with_correct_raw_end(self):
        r = enrich(case(raw_end_correct=True, selected_span_correct=False))
        self.assertTrue(r["lost_correct_raw_end"])
        self.assertFalse(r["recovered_wrong_raw_end"])

    def test_score_margins(self):
        r = enrich(case(raw_end_correct=False, selected_span_correct=True))
        self.assertEqual(r["end_score_margin"], 1.0)
        self.assertEqual(r["gold_joint_score"], 2.0)
        self.assertEqual(r["joint_score_margin"], 2.0)

    def test_aggregate_does_not_drop_recoveries(self):
        rows = [
            enrich(case(raw_end_correct=False, selected_span_correct=True)),
            enrich(case(raw_end_correct=True, selected_span_correct=False)),
        ]
        summary = measure(rows)
        self.assertEqual(summary["exact"], 1)
        self.assertEqual(summary["recovery"], 1)
        self.assertEqual(summary["lost_correct_end"], 1)

    def test_inconsistent_correctness_rejected(self):
        with self.assertRaises(ValueError):
            enrich(case(raw_end_correct=False, selected_span_correct=True, exact=False))


if __name__ == "__main__":
    unittest.main()
