"""Regression tests for V5B fixed-length controls and boundary diagnostics."""

import copy
import io
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from evidence_v3_validation import parse_fields
from evaluate_span_v5b import evaluate_group
from generate_evidence_data_v3 import build_datasets
from generate_span_data_v5b import build_additions, validate_additions
from span_model_v5 import (
    SpanGroundedQA, prepare_span_input, training_span_target,
)
from tokenizer import encode
import train_span_v5b


class VariableSpanV5BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.v3 = build_datasets(seed=42)
        cls.extra = build_additions(seed=42)

    def test_deterministic_partitions_and_lengths(self):
        self.assertEqual(self.extra, build_additions(seed=42))
        counts = {"extra_train": 72, "familiar5": 24, "novel4": 24}
        expected_lengths = {
            "extra_train": {3: 24, 5: 24, 6: 24},
            "familiar5": {5: 24}, "novel4": {4: 24},
        }
        for group, rows in self.extra.items():
            self.assertEqual(len(rows), counts[group])
            validate_additions(group, rows)
            lengths = Counter(
                len(parse_fields(r["information"], r["source_id"])["Leader"])
                for r in rows
            )
            self.assertEqual(dict(lengths), expected_lengths[group])

    def test_name_and_information_partition_isolation(self):
        groups = {**self.v3, **self.extra}
        info = [r["information"] for rows in groups.values() for r in rows]
        ids = [r["source_id"] for rows in groups.values() for r in rows]
        self.assertEqual(len(info), len(set(info)))
        self.assertEqual(len(ids), len(set(ids)))
        v3_names = {
            parse_fields(r["information"], r["source_id"])["Leader"]
            for rows in self.v3.values() for r in rows if r["answerability"] == "answerable"
        }
        for group in ("extra_train", "familiar5", "novel4"):
            names = {
                parse_fields(r["information"], r["source_id"])["Leader"]
                for r in self.extra[group]
            }
            self.assertTrue(v3_names.isdisjoint(names))
        train = {r["answer"] for r in self.extra["extra_train"]}
        self.assertTrue(train.isdisjoint({r["answer"] for r in self.extra["familiar5"]}))
        self.assertTrue(train.isdisjoint({r["answer"] for r in self.extra["novel4"]}))

    def test_modification_of_locked_partition_rejected(self):
        corrupted = copy.deepcopy(self.extra["extra_train"])
        corrupted[0]["answer"] = "Leader: Wrong."
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_additions("extra_train", corrupted)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_additions("train", self.extra["extra_train"])

    def test_all_new_training_targets_exact_source_bytes(self):
        for row in self.extra["extra_train"]:
            token_ids, mask = prepare_span_input(row["information"], row["question"])
            answerable, start, end = training_span_target(row, token_ids, mask)
            self.assertEqual(answerable, 1.0)
            self.assertTrue(mask[0, start:end+1].all().item())
            encoded_name = encode(row["answer"][len("Leader: "):-1])
            self.assertEqual(token_ids[0, start:end+1].tolist(), encoded_name)
            self.assertEqual(len(encoded_name), end-start+1)

    def test_boundary_metrics_differentiate_start_from_end(self):
        row = self.extra["familiar5"][0]
        token_ids, mask = prepare_span_input(row["information"], row["question"])
        _, correct_start, correct_end = training_span_target(row, token_ids, mask)

        class WrongEndModel(SpanGroundedQA):
            def forward(self, tokens, candidate_mask):
                start_logits = torch.full(tokens.shape, -100.0)
                end_logits = torch.full(tokens.shape, -100.0)
                start_logits[0, correct_start] = 10.0
                end_logits[0, correct_end+1] = 10.0
                return start_logits, end_logits, torch.tensor([10.0])

        scores = evaluate_group(WrongEndModel(), [row])
        self.assertEqual(scores["start_head"], 1)
        self.assertEqual(scores["end_head"], 0)
        self.assertEqual(scores["joint_start"], 1)
        self.assertEqual(scores["joint_end"], 0)
        self.assertEqual(scores["joint_span"], 0)
        self.assertEqual(scores["too_long"], 1)
        self.assertEqual(scores["correct"], 0)

    def test_train_script_rejects_evaluation_file(self):
        with patch.object(sys, "argv", [
            "train_span_v5b.py",
            "--experiment-data", "data/evaluation/evidence_sensitivity_v3_known.jsonl",
            "--extra-train", "data/training/evidence_span_v5b_extra_train.jsonl",
            "--checkpoint", "models/do_not_write_v5b.pt",
        ]):
            with redirect_stdout(io.StringIO()) as output:
                status = train_span_v5b.main()
        self.assertEqual(status, 1)
        self.assertIn("only registered V3 TRAINING", output.getvalue())


if __name__ == "__main__":
    unittest.main()
