"""Regression tests for V4 pointer-generator and backward compatibility."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional as F

from grounded_qa.training.checkpoint import load_checkpoint, save_checkpoint
from grounded_qa.models.copy_model import CopyGroundedQA
from grounded_qa.models.model import ModelConfig, TinyGroundedQA
from grounded_qa.tokenization.tokenizer import BYTE_OFFSET, EOS_ID, VOCAB_SIZE, encode


class CopyModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = CopyGroundedQA()
        self.model.eval()
        self.source = torch.tensor(
            [encode("Information:\nLeader: Xezen.\nQuestion:\nWho leads?\nAnswer:\n",
                    add_eos=True)], dtype=torch.long,
        )
        self.decoder = torch.tensor(
            [encode("Leader: X", add_bos=True)], dtype=torch.long,
        )

    def test_normalized_distribution_and_shapes(self):
        with torch.no_grad():
            log_probability = self.model(self.source, self.decoder)
        self.assertEqual(log_probability.shape,
                         (1, self.decoder.shape[1], VOCAB_SIZE))
        self.assertTrue(torch.isfinite(log_probability).all().item())
        self.assertTrue(torch.allclose(
            log_probability.exp().sum(dim=-1),
            torch.ones_like(log_probability[..., 0]), atol=1e-5,
        ))

    def test_source_only_byte_receives_copy_probability(self):
        # The model's source contains X; ordinary vocab generation is still
        # possible. With the gate forced toward copying, EOS (not in copyable
        # positions) becomes negligible; another source byte remains possible.
        with torch.no_grad():
            self.model.generate_gate.weight.zero_()
            self.model.generate_gate.bias.fill_(-25.0)
            probabilities = self.model(self.source, self.decoder).exp()
        self.assertLess(probabilities[0, -1, EOS_ID].item(), 1e-7)
        self.assertGreater(probabilities[0, -1, ord("X") + BYTE_OFFSET].item(), 0)

    def test_padded_source_is_masked_from_copying(self):
        short = self.source[0]
        padded = F.pad(short, (0, 8), value=0).unsqueeze(0)
        with torch.no_grad():
            short_probability = self.model(self.source, self.decoder).exp()
            padded_probability = self.model(padded, self.decoder).exp()
        self.assertTrue(torch.allclose(
            short_probability, padded_probability, atol=1e-5, rtol=1e-4,
        ))

    def test_gradient_reaches_copy_parameters(self):
        self.model.train()
        expected = torch.tensor([[ord("X") + BYTE_OFFSET] * self.decoder.shape[1]],
                                dtype=torch.long)
        logits = self.model(self.source, self.decoder)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), expected.reshape(-1))
        loss.backward()
        self.assertIsNotNone(self.model.copy_query.weight.grad)
        self.assertGreater(self.model.copy_query.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(self.model.generate_gate.weight.grad)
        self.assertGreater(self.model.generate_gate.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_roundtrip_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as root:
            pointer_path = Path(root) / "pointer.pt"
            info = {"steps": 1, "prompt_version": "compact_v1",
                    "model_variant": "pointer_v4"}
            save_checkpoint(pointer_path, self.model, info)
            restored, restored_info = load_checkpoint(pointer_path)
            self.assertIsInstance(restored, CopyGroundedQA)
            self.assertEqual(restored_info["model_variant"], "pointer_v4")
            with torch.no_grad():
                self.assertTrue(torch.allclose(
                    self.model(self.source, self.decoder),
                    restored(self.source, self.decoder),
                    atol=0.0, rtol=0.0,
                ))

            legacy_path = Path(root) / "legacy.pt"
            save_checkpoint(legacy_path, TinyGroundedQA(),
                            {"steps": 1, "prompt_version": "compact_v1"})
            old, legacy_info = load_checkpoint(legacy_path)
            self.assertIsInstance(old, TinyGroundedQA)
            self.assertNotIsInstance(old, CopyGroundedQA)
            self.assertEqual(legacy_info.get("model_variant", "standard"), "standard")


if __name__ == "__main__":
    unittest.main()
