"""Regression tests for the V2B class-aware training sampler."""

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_evidence_data_v2 import build_datasets
from train import class_aware_indices


class EvidenceSamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_datasets(seed=42)["train"]

    def test_2000_updates_uses_requested_class_ratio(self):
        indices = class_aware_indices(self.rows, 2000, 0.33)
        self.assertEqual(len(indices), 2000)
        counts = Counter(self.rows[i]["answerability"] for i in indices)
        self.assertEqual(counts["unanswerable"], 660)
        self.assertEqual(counts["answerable"], 1340)
        self.assertEqual(set(indices), set(range(108)))

    def test_deterministic_for_same_input_and_settings(self):
        self.assertEqual(
            class_aware_indices(self.rows, 100, 0.33),
            class_aware_indices(self.rows, 100, 0.33),
        )

    def test_alternating_classes_use_independent_round_robin(self):
        rows = [
            {"answerability": "answerable"},
            {"answerability": "answerable"},
            {"answerability": "unanswerable"},
        ]
        indices = class_aware_indices(rows, 6, 0.5)
        self.assertEqual(indices, [0, 2, 1, 2, 0, 2])

    def test_rejects_invalid_ratio(self):
        for ratio in (-0.1, 0.0, 1.0, 2.0, float("nan"), float("inf")):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    class_aware_indices(self.rows, 100, ratio)

    def test_requires_both_classes(self):
        for label in ("answerable", "unanswerable"):
            rows = [row for row in self.rows if row["answerability"] == label]
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "requires answerable"):
                    class_aware_indices(rows, 10, 0.33)


if __name__ == "__main__":
    unittest.main()
