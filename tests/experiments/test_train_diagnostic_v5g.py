"""Read-only diagnostics: verify summary math without loading a checkpoint."""
import unittest

from experiments.question_aware_v5g.data_design import build_partition
from experiments.question_aware_v5g.diagnose_train import (
    counts_by_structure, markdown_report,
)
from experiments.question_aware_v5g.evaluate import enriched, summarize


class V5GTrainDiagnosticTests(unittest.TestCase):
    def test_ideal_predictions_report_complete_pairs_and_missing_targets(self):
        rows = build_partition("train")
        predictions = []
        for row in rows:
            answerable = row["expected_value"] is not None
            predictions.append(enriched(row, {
                "predicted_field": row["target_field"] if answerable else None,
                "predicted_value": row["expected_value"],
                "answerability_probability": 0.9 if answerable else 0.1,
                "candidate_fields": [],
                "field_scores": {},
            }))
        metrics = summarize(predictions)
        self.assertEqual(metrics["canonical"]["correct"], 48)
        self.assertEqual(metrics["paired"]["both_correct"], 24)
        self.assertEqual(metrics["paired"]["correct_switches"], 12)
        structure = counts_by_structure(predictions)
        self.assertEqual(sum(s["count"] for s in structure.values()), 48)
        self.assertEqual(set(structure), {
            "Leader+Location+Topic", "Leader+Topic", "Location+Topic",
        })
        report = markdown_report(metrics, None, predictions)
        self.assertIn("48/48 (100.0%)", report)
        self.assertIn("24/24", report)
        self.assertIn("12/12", report)
        self.assertIn("not supplied", report)
        self.assertEqual(report.count("| v5g_train_"), 24)


if __name__ == "__main__":
    unittest.main()
