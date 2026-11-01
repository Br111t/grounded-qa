"""Pure, checkpoint-free tests of V5D boundary-error reporting."""
import unittest

from experiments.balanced_boundary_v5d.diagnose_boundary_errors import (
    format_summary, summarize,
)


def row(*, raw_end_delta, selected_end_delta, raw_end_correct=False,
        selected_span_correct=False, exact=False, abstained=False):
    return {
        "raw_end_delta": raw_end_delta,
        "selected_end_delta": selected_end_delta,
        "raw_end_correct": int(raw_end_correct),
        "raw_start_correct": 1,
        "selected_span_correct": int(selected_span_correct),
        "exact": int(exact),
        "abstained": int(abstained),
    }


class BoundaryDiagnosticTests(unittest.TestCase):
    def test_offset_histograms_and_disagreement(self):
        s = summarize([
            row(raw_end_delta=-1, selected_end_delta=0, selected_span_correct=True),
            row(raw_end_delta=0, selected_end_delta=-1, raw_end_correct=True),
            row(raw_end_delta=2, selected_end_delta=None, abstained=True),
        ])
        self.assertEqual(s["raw_end_deltas"], {"-1": 1, "0": 1, "2": 1})
        self.assertEqual(s["selected_end_deltas"], {"-1": 1, "0": 1})
        self.assertEqual(s["raw_end_correct_but_selected_wrong"], 1)
        self.assertEqual(s["raw_end_wrong_but_selected_correct"], 1)
        self.assertEqual(s["abstentions"], 1)

    def test_empty_records_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])

    def test_exact_and_head_metrics_are_separate(self):
        s = summarize([
            row(raw_end_delta=0, selected_end_delta=-1,
                raw_end_correct=True),
            row(raw_end_delta=-1, selected_end_delta=0,
                selected_span_correct=True, exact=True),
        ])
        self.assertEqual(s["exact"], 1)
        self.assertEqual(s["raw_end_correct"], 1)
        self.assertEqual(s["selected_span_correct"], 1)


if __name__ == "__main__":
    unittest.main()
