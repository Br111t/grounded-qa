"""Tests for evidence-sensitivity V2 generator, labels, partitions, and training guard."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_v2_validation import validate_v2_rows
from generate_evidence_data_v2 import (
    OUTPUTS, QUESTION, TRAIN_NAMES, UNSEEN_NAMES, build_datasets,
)
import train


class EvidenceV2Tests(unittest.TestCase):
    def setUp(self):
        self.data = build_datasets(seed=42)

    def test_counts_fixed_question_and_reference_correctness(self):
        self.assertEqual({key: len(value) for key, value in self.data.items()},
                         {"train": 108, "known": 24, "unseen": 24, "missing": 12})
        for group, rows in self.data.items():
            validate_v2_rows(rows, group=group)
            self.assertTrue(all(row["question"] == QUESTION for row in rows))

    def test_partition_names_and_no_information_overlap(self):
        all_rows = [row for rows in self.data.values() for row in rows]
        self.assertEqual(len({row["information"] for row in all_rows}), len(all_rows))
        train_present = {
            row["answer"].removeprefix("Leader: ").removesuffix(".")
            for row in self.data["train"] if row["answerability"] == "answerable"
        }
        known = {
            row["answer"].removeprefix("Leader: ").removesuffix(".")
            for row in self.data["known"]
        }
        unseen = {
            row["answer"].removeprefix("Leader: ").removesuffix(".")
            for row in self.data["unseen"]
        }
        self.assertEqual(train_present, set(TRAIN_NAMES))
        self.assertEqual(known, set(TRAIN_NAMES))
        self.assertEqual(unseen, set(UNSEEN_NAMES))
        self.assertTrue(train_present.isdisjoint(unseen))
        self.assertTrue(all(row["answerability"] == "unanswerable"
                            for row in self.data["missing"]))

    def test_seed_reproducibility(self):
        self.assertEqual(build_datasets(42), build_datasets(42))
        self.assertNotEqual(build_datasets(42)["train"], build_datasets(43)["train"])

    def test_rejects_wrong_answer_and_question(self):
        examples = [dict(row) for row in self.data["train"]]
        examples[0]["answer"] = "Leader: Incorrect."
        with self.assertRaisesRegex(ValueError, "Answer does not match"):
            validate_v2_rows(examples, group="train")
        examples[0] = dict(self.data["train"][0])
        examples[0]["question"] = "Who leads?"
        with self.assertRaisesRegex(ValueError, "question changed"):
            validate_v2_rows(examples, group="train")

    def test_training_only_loads_v2_train_not_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "evidence_sensitivity_v2_train.jsonl"
            train_path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.data["train"]),
                encoding="utf-8",
            )
            with patch.object(train, "EVIDENCE_V2_TRAIN_PATH", train_path):
                rows = train.load_training_examples(train_path, evidence_experiment=True)
                self.assertEqual(len(rows), 108)
                with self.assertRaisesRegex(ValueError, "Only the registered"):
                    train.load_training_examples(
                        OUTPUTS["unseen"], evidence_experiment=True
                    )


if __name__ == "__main__":
    unittest.main()
