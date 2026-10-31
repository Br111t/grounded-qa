"""Targeted tests for the fixed-question evidence-sensitivity training path."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import grounded_qa.training.train as train


def example(index, leader):
    information = "Fictional Workshop W0001. "
    if leader is not None:
        information += f"Leader: {leader}. "
    information += "Location: Room Maple."
    return {
        "example_id": f"evidence_{index:03d}_q01",
        "source_id": f"evidence_{index:03d}",
        "information": information,
        "question": train.EVIDENCE_QUESTION,
        "answerability": "answerable" if leader else "unanswerable",
        "answer": f"Leader: {leader}." if leader else "Insufficient information.",
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvidenceTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.path = root / "evidence_sensitivity_v1.jsonl"
        self.rows = [example(1, "Mira"), example(2, "Tavi"), example(3, None)]
        write_jsonl(self.path, self.rows)
        patcher = patch.object(train, "EVIDENCE_EXPERIMENT_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_evidence_experiment_loads_all_variants(self):
        loaded = train.load_training_examples(self.path, evidence_experiment=True)
        self.assertEqual(loaded, self.rows)
        self.assertEqual(len(loaded), 3)

    def test_rejects_other_path_without_reading(self):
        other = self.path.parent / "holdout.jsonl"
        with self.assertRaisesRegex(ValueError, "Only the registered"):
            train.load_training_examples(other, evidence_experiment=True)

    def test_rejects_question_changes(self):
        changed = [dict(row) for row in self.rows]
        changed[0]["question"] = "Where is the workshop?"
        write_jsonl(self.path, changed)
        with self.assertRaisesRegex(ValueError, "identical fixed question"):
            train.load_training_examples(self.path, evidence_experiment=True)

    def test_rejects_incorrect_answer(self):
        changed = [dict(row) for row in self.rows]
        changed[1]["answer"] = "Leader: Mira."
        write_jsonl(self.path, changed)
        with self.assertRaisesRegex(ValueError, "does not support"):
            train.load_training_examples(self.path, evidence_experiment=True)

    def test_rejects_extra_fact_variation(self):
        changed = [dict(row) for row in self.rows]
        changed[1]["information"] = changed[1]["information"].replace("Room Maple", "Room Cedar")
        write_jsonl(self.path, changed)
        with self.assertRaisesRegex(ValueError, "Unexpected evidence format"):
            train.load_training_examples(self.path, evidence_experiment=True)

    def test_original_source_allowlist_is_unchanged(self):
        original = self.path.parent / "generated_train.jsonl"
        write_jsonl(original, [example(9, "Ari")])
        with patch.object(train, "TRAIN_PATH", original):
            with self.assertRaisesRegex(ValueError, "not in the original training split"):
                train.load_training_examples(self.path)


if __name__ == "__main__":
    unittest.main()
