"""Read-only V4 copy tracing tests, independent of trained weights."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from grounded_qa.inference.check_model import trace_copy_details
from grounded_qa.models.copy_model import CopyGroundedQA
from grounded_qa.prompting.prompting import build_prompt
from grounded_qa.tokenization.tokenizer import BYTE_OFFSET, VOCAB_SIZE, encode


class CopyDetailsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_components_recompose_forward_without_changing_predictions(self):
        torch.manual_seed(42)
        model = CopyGroundedQA().eval()
        prompt = build_prompt(
            "Fictional Workshop W0001. Leader: Xezen. Location: Room Maple.",
            "What is the leader of Workshop W0001?",
            version="compact_v1",
        )
        source = torch.tensor([encode(prompt, add_eos=True)], dtype=torch.long)
        decoder = torch.tensor([encode("Leader: X", add_bos=True)], dtype=torch.long)
        weights_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        with torch.no_grad():
            normal = model(source, decoder)
            diagnostic, parts = model.inspect_copy(source, decoder)

        self.assertTrue(torch.equal(normal, diagnostic))
        self.assertEqual(parts["copy_attention"].shape, (1, decoder.shape[1], source.shape[1]))
        self.assertEqual(parts["copy_probability"].shape, (1, decoder.shape[1], VOCAB_SIZE))
        g = parts["generate_fraction"]
        composed = (g * parts["vocabulary_probability"]
                    + (1-g) * parts["copy_probability"])
        self.assertTrue(torch.allclose(
            composed, parts["combined_probability"], atol=1e-7, rtol=1e-6
        ))
        self.assertTrue(torch.allclose(
            diagnostic.exp(), composed, atol=1e-6, rtol=1e-5
        ))
        self.assertTrue(torch.allclose(
            parts["copy_attention"].sum(-1),
            torch.ones_like(parts["copy_attention"][..., 0]),
            atol=1e-6,
        ))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, weights_before[name]))

    def test_copy_inspection_requires_eval(self):
        model = CopyGroundedQA()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            model.inspect_copy(torch.empty((1, 1), dtype=torch.long),
                               torch.empty((1, 1), dtype=torch.long))

    def test_trace_uses_generated_prefix_and_reports_name_byte_measures(self):
        # A deterministic fake controls ONLY the diagnostic's generation route.
        # The reference answer is deliberately different: it must not be
        # supplied to the decoder at any point.
        class PredictablePointer:
            config = SimpleNamespace(max_target_tokens=32)

            def inspect_copy(self, source, decoder):
                expected_ids = encode("Leader: Xezen.", add_eos=True)
                step = decoder.shape[1] - 1
                target = expected_ids[step]
                length = decoder.shape[1]
                logits = torch.full((1, length, VOCAB_SIZE), -30.0)
                logits[0, -1, target] = 0.0
                vocabulary = torch.zeros_like(logits)
                vocabulary[0, -1, target] = 1.0
                copying = torch.zeros_like(logits)
                copying[0, -1, target] = 1.0
                attention = torch.full(
                    (1, length, source.shape[1]), 1 / source.shape[1]
                )
                gate = torch.full((1, length, 1), 0.75)
                return logits, {
                    "generate_fraction": gate,
                    "vocabulary_probability": vocabulary,
                    "copy_probability": copying,
                    "copy_attention": attention,
                    "combined_probability": 0.75 * vocabulary + 0.25 * copying,
                }

        output = io.StringIO()
        with redirect_stdout(output):
            result = trace_copy_details(
                PredictablePointer(),
                "Fictional Workshop W0001. Leader: Xezen. Location: Room Maple.",
                "What is the leader of Workshop W0001?",
                "Leader: Incorrect.",
                prompt_version="compact_v1",
            )
        self.assertEqual(result, "Leader: Xezen.")
        text = output.getvalue()
        self.assertIn("free-running greedy output", text)
        self.assertIn("generate=0.750, copy=0.250", text)
        self.assertIn("P_vocab=1.000, P_copy=1.000", text)
        self.assertIn("attention on matching leader bytes=", text)
        self.assertIn("Reference (comparison only): 'Leader: Incorrect.'", text)


if __name__ == "__main__":
    unittest.main()
