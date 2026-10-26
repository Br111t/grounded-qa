"""Check checkpoint preservation and compatibility without running training."""

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import checkpoint
from model import ModelConfig, TinyGroundedQA
from tokenizer import BOS_ID, EOS_ID
import train


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix=".checkpoint-test-", dir=PROJECT_ROOT))
        self.addCleanup(self.remove_test_directory)
        torch.manual_seed(42)
        self.config = ModelConfig(
            d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=16, dropout=0.0, max_input_tokens=16, max_target_tokens=8,
        )
        self.model = TinyGroundedQA(self.config).eval()
        self.metadata = {"steps": 3, "seed": 42, "example_ids": ["test_example"]}
        self.path = self.directory / "original.pt"
        checkpoint.save_checkpoint(self.path, self.model, self.metadata)

    def remove_test_directory(self):
        # Resolve and verify the target before recursive cleanup on Windows.
        resolved = self.directory.resolve()
        if resolved == PROJECT_ROOT or not resolved.is_relative_to(PROJECT_ROOT):
            raise RuntimeError(f"Refusing cleanup outside the project: {resolved}")
        shutil.rmtree(resolved)

    def test_nondefault_config_round_trip_preserves_weights_and_logits(self):
        restored, metadata = checkpoint.load_checkpoint(self.path)
        self.assertEqual(restored.config, self.config)
        self.assertEqual(metadata, self.metadata)
        self.assertFalse(restored.training)
        self.assertEqual(set(restored.state_dict()), set(self.model.state_dict()))
        for name, actual in restored.state_dict().items():
            with self.subTest(weight=name):
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, self.model.state_dict()[name]))
        source = torch.tensor([[10, 11, EOS_ID]], dtype=torch.long)
        decoder = torch.tensor([[BOS_ID, 12, 13]], dtype=torch.long)
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(source, decoder), restored(source, decoder)))

    def test_existing_checkpoint_is_not_overwritten(self):
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            checkpoint.save_checkpoint(self.path, self.model, self.metadata)
        self.assertEqual(self.path.read_bytes(), original)

    def test_failed_new_write_removes_only_its_partial_file(self):
        original = self.path.read_bytes()
        incomplete = self.directory / "incomplete.pt"

        def fail_after_writing(_payload, stream):
            stream.write(b"partial checkpoint")
            raise OSError("simulated write failure")

        with mock.patch.object(checkpoint.torch, "save", side_effect=fail_after_writing):
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                checkpoint.save_checkpoint(incomplete, self.model, self.metadata)
        self.assertFalse(incomplete.exists())
        self.assertEqual(self.path.read_bytes(), original)

    def test_incompatible_metadata_and_weights_are_rejected(self):
        original = torch.load(self.path, map_location="cpu", weights_only=True)
        weight = next(iter(original["model_state_dict"]))
        mutations = {
            "version": lambda p: p.update(format_version=999),
            "tokenizer": lambda p: p["tokenizer"].update(vocab_size=999),
            "prompt": lambda p: p.update(prompt_template="different prompt"),
            "missing_config": lambda p: p.pop("model_config"),
            "missing_config_field": lambda p: p["model_config"].pop("d_model"),
            "missing_weight": lambda p: p["model_state_dict"].pop(weight),
            "extra_weight": lambda p: p["model_state_dict"].update(extra=torch.zeros(1)),
            "misshaped_weight": lambda p: p["model_state_dict"].update(
                {weight: p["model_state_dict"][weight][:-1]}
            ),
            "wrong_dtype": lambda p: p["model_state_dict"].update(
                {weight: p["model_state_dict"][weight].double()}
            ),
            "nonfinite_weight": lambda p: p["model_state_dict"][weight].fill_(float("nan")),
        }
        for name, mutate in mutations.items():
            with self.subTest(problem=name):
                payload = copy.deepcopy(original)
                mutate(payload)
                invalid_path = self.directory / f"{name}.pt"
                torch.save(payload, invalid_path)
                with self.assertRaises((ValueError, RuntimeError)):
                    checkpoint.load_checkpoint(invalid_path)

    def test_truncated_checkpoint_is_rejected(self):
        truncated = self.directory / "truncated.pt"
        truncated.write_bytes(self.path.read_bytes()[:30])
        with self.assertRaises(ValueError):
            checkpoint.load_checkpoint(truncated)

    def test_train_refuses_existing_path_before_reading_data_or_building_model(self):
        original = self.path.read_bytes()
        arguments = ["train.py", "--checkpoint", str(self.path), "--steps", "1"]
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(train, "TinyGroundedQA") as build_model,
            mock.patch.object(train, "validate_data") as read_data,
            redirect_stdout(output),
        ):
            self.assertEqual(train.main(), 1)
        build_model.assert_not_called()
        read_data.assert_not_called()
        self.assertIn("already exists", output.getvalue())
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
