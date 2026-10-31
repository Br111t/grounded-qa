"""Regression tests for versioned prompt formats and legacy checkpoints."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from grounded_qa.training.checkpoint import PROMPT_TEMPLATES, load_checkpoint, save_checkpoint
from grounded_qa.models.model import TinyGroundedQA
from grounded_qa.prompting.prompting import PROMPT_VERSIONS, build_prompt
from grounded_qa.training.train import prepare_example
from grounded_qa.tokenization.tokenizer import decode


class PromptVersionTests(unittest.TestCase):
    def test_original_is_unchanged(self):
        original = build_prompt("Leader: Mira.", "Who leads?")
        self.assertEqual(original, build_prompt("Leader: Mira.", "Who leads?", version="original"))
        self.assertIn("Instructions:\n", original)
        self.assertEqual(PROMPT_TEMPLATES["original"], build_prompt("{information}", "{question}"))

    def test_compact_omits_long_instructions(self):
        expected = "Information:\nLeader: Mira.\n\nQuestion:\nWho leads?\n\nAnswer:\n"
        self.assertEqual(build_prompt("Leader: Mira.", "Who leads?", version="compact_v1"), expected)
        self.assertNotIn("Instructions:", expected)

    def test_version_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            build_prompt("Leader: Mira.", "Who leads?", version="unknown")

    def test_training_prepares_compact_input_without_changing_target(self):
        row = {"information": "Leader: Mira.", "question": "Who leads?", "answer": "Leader: Mira."}
        compact_source, compact_decoder, compact_label = prepare_example(row, prompt_version="compact_v1")
        original_source, original_decoder, original_label = prepare_example(row)
        self.assertLess(compact_source.shape[1], original_source.shape[1])
        self.assertTrue(torch.equal(compact_decoder, original_decoder))
        self.assertTrue(torch.equal(compact_label, original_label))

    def test_checkpoint_roundtrips_for_both_versions(self):
        model = TinyGroundedQA()
        with tempfile.TemporaryDirectory() as tmp:
            for version in PROMPT_VERSIONS:
                filename = Path(tmp) / f"{version}.pt"
                training_info = {"steps": 1}
                if version == "compact_v1":
                    training_info["prompt_version"] = version
                save_checkpoint(filename, model, training_info)
                loaded, info = load_checkpoint(filename)
                self.assertEqual(info.get("prompt_version", "original"), version)
                self.assertEqual(loaded.config, model.config)


if __name__ == "__main__":
    unittest.main()
