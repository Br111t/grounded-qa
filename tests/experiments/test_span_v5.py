"""V5 extraction, train-only labels, gradients and checkpoint regression tests."""

import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import build_datasets
from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.models.span_model_v5 import (
    SpanGroundedQA, load_span_checkpoint, predict_span, prepare_span_input,
    save_span_checkpoint, span_training_loss, training_span_target,
)
from grounded_qa.tokenization.tokenizer import BOS_ID, BYTE_OFFSET, encode
import experiments.span_extractor_v5.train_span_v5 as train_span_v5


class SpanV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.groups = build_datasets(seed=42)
        cls.answerable = next(
            row for row in cls.groups["train"] if row["answerability"] == "answerable"
        )
        cls.missing = next(
            row for row in cls.groups["train"] if row["answerability"] == "unanswerable"
        )

    def test_evidence_mask_excludes_question_and_prompt(self):
        row = self.answerable
        source, mask = prepare_span_input(row["information"], row["question"])
        self.assertEqual(source[0, 0].item(), BOS_ID)
        self.assertFalse(mask[0, 0].item())
        self.assertEqual(int(mask.sum().item()), len(encode(row["information"])))
        self.assertTrue(mask[0, -1].item() is False)
        prompt = build_prompt(row["information"], row["question"], version="compact_v1")
        self.assertEqual(
            len(source[0]), len(encode(prompt, add_eos=True)) + 1
        )
        self.assertEqual(int(source[0, -1].item()), 2)

    def test_all_training_targets_refer_to_real_evidence_bytes(self):
        for row in self.groups["train"]:
            source, mask = prepare_span_input(row["information"], row["question"])
            answerable, start, end = training_span_target(row, source, mask)
            if row["answerability"] == "unanswerable":
                self.assertEqual((answerable, start, end), (0.0, None, None))
            else:
                self.assertEqual(answerable, 1.0)
                self.assertTrue(mask[0, start:end+1].all().item())
                selected = bytes(source[0, start:end+1].tolist()[i] - BYTE_OFFSET
                                 for i in range(end-start+1))
                self.assertEqual(
                    selected.decode("utf-8"), row["answer"][len("Leader: "):-1]
                )

    def test_wrong_reference_rejected_before_training(self):
        row = copy.deepcopy(self.answerable)
        row["answer"] = "Leader: Wrong."
        source, mask = prepare_span_input(row["information"], row["question"])
        with self.assertRaisesRegex(ValueError, "reference"):
            training_span_target(row, source, mask)

    def test_missing_row_has_only_answerability_loss(self):
        torch.manual_seed(42)
        model = SpanGroundedQA().train()
        row = self.missing
        source, mask = prepare_span_input(row["information"], row["question"])
        target = training_span_target(row, source, mask)
        total, classification, extraction = span_training_loss(model, source, mask, target)
        self.assertTrue(torch.equal(total, classification))
        self.assertEqual(extraction.item(), 0)
        total.backward()
        self.assertGreater(model.answerability_head.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.start_head.weight.grad)
        self.assertIsNone(model.end_head.weight.grad)

    def test_answerable_gradient_reaches_span_heads(self):
        torch.manual_seed(42)
        model = SpanGroundedQA().train()
        row = self.answerable
        source, mask = prepare_span_input(row["information"], row["question"])
        target = training_span_target(row, source, mask)
        total, classification, extraction = span_training_loss(model, source, mask, target)
        self.assertGreater(extraction.item(), 0)
        self.assertGreater(classification.item(), 0)
        total.backward()
        self.assertGreater(model.start_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.end_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.answerability_head.weight.grad.abs().sum().item(), 0)

    def test_extracts_unseen_span_when_span_heads_identify_it(self):
        """Validates decoding/selection only, NOT a trained model's accuracy."""
        example = self.groups["unseen"][0]
        name = example["answer"][len("Leader: "):-1]
        class Scripted(SpanGroundedQA):
            def __init__(self, answerable=True):
                super().__init__()
                self.answerable = answerable

            def forward(self, token_ids, candidate_mask):
                bytes_to_find = encode(name)
                tokens = token_ids[0].tolist()
                matches = [i for i in range(len(tokens)-len(bytes_to_find)+1)
                           if tokens[i:i+len(bytes_to_find)] == bytes_to_find
                           and candidate_mask[0, i:i+len(bytes_to_find)].all().item()]
                if len(matches) != 1:
                    raise AssertionError("Expected a unique name in Information.")
                start = torch.full(token_ids.shape, -100.0)
                end = torch.full(token_ids.shape, -100.0)
                start[0, matches[0]] = 100.0
                end[0, matches[0] + len(bytes_to_find) - 1] = 100.0
                answerable = torch.tensor([10.0 if self.answerable else -10.0])
                return start, end, answerable

        predicted, prob, span = predict_span(
            Scripted(), example["information"], example["question"]
        )
        self.assertEqual(predicted, example["answer"])
        self.assertIsNotNone(span)
        self.assertGreater(prob, 0.5)
        absent, prob, span = predict_span(
            Scripted(answerable=False), example["information"], example["question"]
        )
        self.assertEqual(absent, "Insufficient information.")
        self.assertIsNone(span)

    def test_checkpoint_roundtrip_and_cannot_overwrite(self):
        torch.manual_seed(42)
        model = SpanGroundedQA().eval()
        source, mask = prepare_span_input(
            self.answerable["information"], self.answerable["question"]
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "test_span.pt"
            save_span_checkpoint(path, model, {"steps": 1, "training_file": "training"})
            loaded, meta = load_span_checkpoint(path)
            self.assertIsInstance(loaded, SpanGroundedQA)
            self.assertEqual(meta["steps"], 1)
            with torch.no_grad():
                for old, new in zip(model(source, mask), loaded(source, mask)):
                    self.assertTrue(torch.equal(old, new))
            with self.assertRaisesRegex(ValueError, "overwriting"):
                save_span_checkpoint(path, model, {"steps": 1})

    def test_training_rejects_evaluation_data_path(self):
        with patch.object(sys, "argv", [
            "train_span_v5.py",
            "--experiment-data", "data/evaluation/evidence_sensitivity_v3_unseen.jsonl",
            "--checkpoint", "models/never_write_v5.pt",
        ]):
            with redirect_stdout(io.StringIO()) as output:
                status = train_span_v5.main()
        self.assertEqual(status, 1)
        self.assertIn("only on the registered V3 TRAINING split", output.getvalue())


if __name__ == "__main__":
    unittest.main()
