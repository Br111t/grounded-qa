"""V4B training-only alignment and gate-supervision regression tests."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional as F

from grounded_qa.models.copy_model import CopyGroundedQA
from experiments.evidence_copy_v4b.evidence_copy_v4b import auxiliary_copy_losses, derive_name_alignment
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import build_datasets
from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.tokenization.tokenizer import VOCAB_SIZE, encode
from grounded_qa.training.train import prepare_example


class EvidenceCopyV4BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.dataset = build_datasets(seed=42)["train"]
        cls.answerable = next(row for row in cls.dataset if row["answerability"] == "answerable")
        cls.missing = next(row for row in cls.dataset if row["answerability"] == "unanswerable")

    def test_training_only_alignment_matches_actual_source_bytes(self):
        for row in self.dataset:
            alignment = derive_name_alignment(row, prompt_version="compact_v1")
            if row["answerability"] == "unanswerable":
                self.assertIsNone(alignment)
                continue
            self.assertIsNotNone(alignment)
            prompt = build_prompt(row["information"], row["question"], version="compact_v1")
            source = encode(prompt, add_eos=True)
            target = encode(row["answer"], add_eos=True)
            name = row["answer"][len("Leader: "):-1]
            self.assertEqual(len(alignment.output_positions), len(encode(name)))
            for target_index, source_index in zip(
                alignment.output_positions, alignment.source_positions
            ):
                self.assertEqual(source[source_index], target[target_index])
            selected = bytes(source[pos] - 3 for pos in alignment.source_positions)
            self.assertEqual(selected.decode("utf-8"), name)

    def test_no_alignment_for_missing_cases(self):
        self.assertIsNone(derive_name_alignment(self.missing, prompt_version="compact_v1"))

    def test_corrupt_answers_and_prompts_are_rejected(self):
        bad = copy.deepcopy(self.answerable)
        bad["answer"] = "Leader: Wrong."
        with self.assertRaisesRegex(ValueError, "does not match"):
            derive_name_alignment(bad, prompt_version="compact_v1")
        with self.assertRaisesRegex(ValueError, "compact_v1"):
            derive_name_alignment(self.answerable, prompt_version="original")

    def test_losses_are_independent_and_gradients_flow(self):
        alignment = derive_name_alignment(self.answerable, prompt_version="compact_v1")
        outlen = max(alignment.output_positions) + 1
        srclen = max(alignment.source_positions) + 1
        attention_logits = torch.randn(1, outlen, srclen, requires_grad=True)
        raw_gate = torch.zeros(1, outlen, 1, requires_grad=True)
        parts = {
            "copy_attention": F.softmax(attention_logits, dim=-1),
            "generate_fraction": torch.sigmoid(raw_gate),
        }

        a, g = auxiliary_copy_losses(
            parts, alignment, alignment_weight=0.5, gate_weight=0.0,
        )
        self.assertGreater(a.item(), 0)
        self.assertEqual(g.item(), 0)
        a.backward(retain_graph=True)
        self.assertGreater(attention_logits.grad.abs().sum().item(), 0)
        # Unused graph branches have no .grad tensor, not necessarily a zero tensor.
        self.assertIsNone(raw_gate.grad)

        # Set gradients to None between checks so we detect actual graph usage.
        attention_logits.grad = None
        raw_gate.grad = None
        a, g = auxiliary_copy_losses(
            parts, alignment, alignment_weight=0.0, gate_weight=0.5,
        )
        self.assertEqual(a.item(), 0)
        self.assertGreater(g.item(), 0)
        g.backward(retain_graph=True)
        self.assertIsNone(attention_logits.grad)
        self.assertGreater(raw_gate.grad.abs().sum().item(), 0)

        attention_logits.grad = None
        raw_gate.grad = None
        a, g = auxiliary_copy_losses(
            parts, alignment, alignment_weight=0.5, gate_weight=0.5,
        )
        (0.5 * a + 0.5 * g).backward()
        self.assertGreater(attention_logits.grad.abs().sum().item(), 0)
        self.assertGreater(raw_gate.grad.abs().sum().item(), 0)

    def test_missing_rows_receive_no_auxiliary_loss(self):
        attention = torch.full((1, 24, 100), 0.01, requires_grad=True)
        gate = torch.full((1, 24, 1), 0.5, requires_grad=True)
        a, g = auxiliary_copy_losses(
            {"copy_attention": attention, "generate_fraction": gate},
            None, alignment_weight=0.5, gate_weight=0.5,
        )
        self.assertEqual(a.item(), 0)
        self.assertEqual(g.item(), 0)

    def test_forward_components_match_original_outputs_and_support_training(self):
        torch.manual_seed(42)
        model = CopyGroundedQA().train()
        source, decoder, labels = prepare_example(
            self.answerable, prompt_version="compact_v1"
        )
        model.eval()
        with torch.no_grad():
            normal = model(source, decoder)
            diagnostics, _ = model.forward_with_components(source, decoder)
        self.assertTrue(torch.equal(normal, diagnostics))

        model.train()
        logits, parts = model.forward_with_components(source, decoder)
        alignment = derive_name_alignment(self.answerable, prompt_version="compact_v1")
        answer_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1)
        )
        a, g = auxiliary_copy_losses(
            parts, alignment, alignment_weight=0.5, gate_weight=0.5
        )
        total = answer_loss + 0.5*a + 0.5*g
        self.assertTrue(torch.isfinite(total).item())
        total.backward()
        self.assertGreater(model.copy_query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.generate_gate.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
