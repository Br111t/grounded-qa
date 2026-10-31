"""Regression checks for fixed-question Evidence Sensitivity V3."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.evidence_sensitivity_v3.evidence_v3_validation import validate_v3_rows
from experiments.evidence_sensitivity_v3.evaluate_evidence_v3 import read_partitions
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import build_datasets
import grounded_qa.training.train as train


class EvidenceV3Tests(unittest.TestCase):
    def setUp(self):
        self.parts = build_datasets(seed=42)

    def test_partition_counts_labels_and_fixed_question(self):
        self.assertEqual(
            {group: len(rows) for group, rows in self.parts.items()},
            {"train": 168, "known": 48, "recombined": 24, "unseen": 24, "missing": 12},
        )
        for group, rows in self.parts.items():
            with self.subTest(group=group):
                validate_v3_rows(rows, group=group)

    def test_seed_reproducibility(self):
        self.assertEqual(self.parts, build_datasets(42))
        self.assertNotEqual(self.parts, build_datasets(43))

    def test_names_and_information_do_not_leak(self):
        names = {}
        all_rows = []
        for group, rows in self.parts.items():
            all_rows.extend(rows)
            names[group] = {
                row["answer"].removeprefix("Leader: ").removesuffix(".")
                for row in rows if row["answerability"] == "answerable"
            }
        self.assertEqual(names["known"], names["train"])
        self.assertFalse(names["train"] & names["recombined"])
        self.assertFalse(names["train"] & names["unseen"])
        self.assertFalse(names["recombined"] & names["unseen"])
        self.assertEqual(len({row["information"] for row in all_rows}), len(all_rows))
        self.assertEqual(len({row["source_id"] for row in all_rows}), len(all_rows))

    def test_rejects_wrong_reference_and_question(self):
        modified = [dict(row) for row in self.parts["train"]]
        modified[0]["answer"] = "Leader: Wrong."
        with self.assertRaisesRegex(ValueError, "Incorrect label or expected answer"):
            validate_v3_rows(modified, "train")
        modified = [dict(row) for row in self.parts["train"]]
        modified[0]["question"] = "Who leads?"
        with self.assertRaisesRegex(ValueError, "question is not fixed"):
            validate_v3_rows(modified, "train")

    def test_training_guard_only_accepts_v3_training(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "evidence_sensitivity_v3_train.jsonl"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.parts["train"]),
                encoding="utf-8",
            )
            with patch.object(train, "EVIDENCE_V3_TRAIN_PATH", train_path):
                loaded = train.load_training_examples(
                    train_path, evidence_experiment=True
                )
                self.assertEqual(loaded, self.parts["train"])
                with self.assertRaisesRegex(ValueError, "Only the registered"):
                    train.load_training_examples(
                        Path(directory) / "evidence_sensitivity_v3_unseen.jsonl",
                        evidence_experiment=True,
                    )

    def test_balanced_sampler_is_deterministic_and_covers_every_row(self):
        indices = train.class_aware_indices(self.parts["train"], 3000, 0.33)
        self.assertEqual(len(indices), 3000)
        self.assertEqual(set(indices), set(range(168)))
        counts = Counter(self.parts["train"][index]["answerability"] for index in indices)
        self.assertEqual(counts["unanswerable"], 990)
        self.assertEqual(counts["answerable"], 2010)
        self.assertEqual(indices, train.class_aware_indices(self.parts["train"], 3000, 0.33))


if __name__ == "__main__":
    unittest.main()
