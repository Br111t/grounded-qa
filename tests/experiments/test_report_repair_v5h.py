"""Report-recovery regression tests: no checkpoint or training artifacts rewritten."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.question_aware_v5h.train_paired_ranking import repair_summary


def example_report(checkpoint):
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    snap = {
        "losses": {
            "answerability_bce_all": 0.65,
            "start_ce_answerable": 4.88,
            "end_ce_answerable": 2.37,
        },
        "correct": 24, "leader_correct": 6, "location_correct": 18,
        "paired_both_correct": 0, "correct_switches": 0,
        "false_abstentions": 0, "unsupported_answers": 12,
        "brier_score": 0.21,
    }
    return {
        "metadata": {
            "experiment_variant": "v5h_paired_ranking",
            "steps": 1200, "seed": 42, "learning_rate": 0.001,
            "rank_weight": 0.5, "rank_margin": 1.0,
            "parameter_count": 100, "training_sha256": "registered-data-sha",
        },
        "checkpoint_sha256": checkpoint_sha,
        "exposure_counts": {"leader/answerable": 450, "location/answerable": 450},
        "runtime": {
            "elapsed_seconds": 10.0, "optimization_seconds": 8.0,
            "ranking_eligible_updates": 600,
        },
        "baseline": {"dev": {
            "correct": 24, "leader_correct": 6, "location_correct": 18,
            "paired_both_correct": 0, "correct_switches": 0,
            "false_abstentions": 0, "unsupported_answers": 12,
            # Intentionally missing brier_score, as in real V5G uniform JSON.
        }},
        "measurements": [{"step": 1200, "train": snap, "dev": snap}],
    }


class V5HReportRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"immutable sample checkpoint")
        self.report_dir = self.root / "report"
        self.report_dir.mkdir()
        self.data = example_report(self.checkpoint)
        self.metrics = self.report_dir / "metrics.json"
        self.metrics.write_text(json.dumps(self.data), encoding="utf-8")
        self.original_metrics = self.metrics.read_bytes()
        self.original_checkpoint = self.checkpoint.read_bytes()

    def test_recovers_missing_summary_with_missing_baseline_brier(self):
        repair_summary(self.checkpoint, self.report_dir)
        summary = (self.report_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn("| brier_score | not recorded | 0.21 |", summary)
        self.assertIn("| correct | 24 | 24 |", summary)
        self.assertEqual(self.metrics.read_bytes(), self.original_metrics)
        self.assertEqual(self.checkpoint.read_bytes(), self.original_checkpoint)

    def test_refuses_to_overwrite_summary(self):
        summary = self.report_dir / "summary.md"
        summary.write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            repair_summary(self.checkpoint, self.report_dir)
        self.assertEqual(summary.read_text(encoding="utf-8"), "keep me")

    def test_rejects_mismatched_checkpoint_sha(self):
        self.checkpoint.write_bytes(b"wrong model")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash disagrees"):
            repair_summary(self.checkpoint, self.report_dir)
        self.assertFalse((self.report_dir / "summary.md").exists())

    def test_rejects_incomplete_training_history(self):
        self.data["measurements"][-1]["step"] = 1080
        self.metrics.write_text(json.dumps(self.data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completed final step"):
            repair_summary(self.checkpoint, self.report_dir)
        self.assertFalse((self.report_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
