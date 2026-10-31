"""Check alternate training datasets without training a Transformer or saving files."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.data.expand_practice import expand_examples
from scripts.data.generate_data import generate_examples
from grounded_qa.tokenization.tokenizer import VOCAB_SIZE
import grounded_qa.training.train as train


SOURCE = "generated_workshop_0001"
OTHER_SOURCE = "generated_workshop_0002"
HELDOUT_SOURCE = "generated_workshop_0003"
PRACTICE_PATH = Path("data/training/reviewed-test-wordings.jsonl")


class FakeModel(nn.Module):
    """Differentiable scores for CLI wiring; the optimizer never updates them."""

    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, source, decoder):
        return self.scores.view(1, 1, -1).expand(source.shape[0], decoder.shape[1], -1)


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        generated = generate_examples(records=3, seed=42)
        self.training_rows = [row for row in generated if row["source_id"] != HELDOUT_SOURCE]
        self.heldout_rows = [row for row in generated if row["source_id"] == HELDOUT_SOURCE]
        self.expanded_rows = expand_examples(self.training_rows, SOURCE)

    def test_default_dataset_is_read_once_and_preserves_rows(self):
        with mock.patch.object(train, "validate_data", return_value=(self.training_rows, [])) as read:
            actual = train.load_training_examples(train.TRAIN_PATH)
        self.assertEqual(actual, self.training_rows)
        read.assert_called_once_with(train.TRAIN_PATH)

    def test_expanded_dataset_keeps_all_twenty_rows_and_their_order(self):
        with mock.patch.object(
            train, "validate_data", side_effect=((self.expanded_rows, []), (self.training_rows, [])),
        ) as read:
            actual = train.load_training_examples(PRACTICE_PATH)
        self.assertEqual(read.call_args_list, [mock.call(PRACTICE_PATH), mock.call(train.TRAIN_PATH)])
        self.assertEqual(actual, self.expanded_rows)
        selected = train.select_examples(actual, source_id=SOURCE)
        self.assertEqual(len(selected), 20)
        self.assertEqual([row["example_id"] for row in selected], [
            f"{SOURCE}_q{number:02d}{suffix}"
            for number in range(1, 11) for suffix in ("", "_alt")
        ])

    def test_disallowed_rows_are_rejected_before_selection_model_and_optimizer(self):
        unknown = {**self.heldout_rows[0], "source_id": "unknown_source"}
        other = next(row for row in self.training_rows if row["source_id"] == OTHER_SOURCE)
        changed = {**other, "information": other["information"] + " Date: 2031-01-01."}
        for reason, extra_row in (
            ("heldout source", self.heldout_rows[0]),
            ("unknown source", unknown),
            ("changed facts", changed),
        ):
            with self.subTest(reason=reason):
                # Each disallowed row lies outside --source-id. It must still
                # be rejected rather than silently discarded by selection.
                requested_rows = self.expanded_rows + [extra_row]
                arguments = [
                    "train.py", "--data", str(PRACTICE_PATH),
                    "--source-id", SOURCE, "--steps", "1",
                ]
                with (
                    mock.patch.object(sys, "argv", arguments),
                    mock.patch.object(train, "validate_data", side_effect=(
                        (requested_rows, []), (self.training_rows, []),
                    )) as read,
                    mock.patch.object(train, "select_examples") as select,
                    mock.patch.object(train, "TinyGroundedQA") as model,
                    mock.patch.object(train.torch.optim, "Adam") as optimizer,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(train.main(), 1)
                self.assertEqual(read.call_args_list, [mock.call(PRACTICE_PATH), mock.call(train.TRAIN_PATH)])
                select.assert_not_called()
                model.assert_not_called()
                optimizer.assert_not_called()

    def test_unknown_source_loader_raises_value_error(self):
        with mock.patch.object(train, "validate_data", side_effect=(
            (self.heldout_rows, []), (self.training_rows, []),
        )):
            with self.assertRaises(ValueError):
                train.load_training_examples(PRACTICE_PATH)

    def test_invalid_requested_data_or_original_split_stops_before_training(self):
        cases = (
            ("requested file", (([], ["invalid requested data"]),), [mock.call(PRACTICE_PATH)]),
            ("original split", (
                (self.expanded_rows, []), ([], ["invalid original split"]),
            ), [mock.call(PRACTICE_PATH), mock.call(train.TRAIN_PATH)]),
        )
        for reason, results, expected_reads in cases:
            with self.subTest(reason=reason):
                with (
                    mock.patch.object(sys, "argv", ["train.py", "--data", str(PRACTICE_PATH), "--steps", "1"]),
                    mock.patch.object(train, "validate_data", side_effect=results) as read,
                    mock.patch.object(train, "select_examples") as select,
                    mock.patch.object(train, "TinyGroundedQA") as model,
                    mock.patch.object(train.torch.optim, "Adam") as optimizer,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(train.main(), 1)
                self.assertEqual(read.call_args_list, expected_reads)
                select.assert_not_called()
                model.assert_not_called()
                optimizer.assert_not_called()

    def test_cli_records_expanded_dataset_path_count_and_ordered_ids(self):
        model = FakeModel()
        original_scores = model.scores.detach().clone()
        destination = Path("unused-expanded-test-checkpoint.pt")
        output = io.StringIO()
        arguments = [
            "train.py", "--data", str(PRACTICE_PATH), "--source-id", SOURCE,
            "--steps", "1", "--checkpoint", str(destination),
        ]
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(train, "validate_data", side_effect=(
                (self.expanded_rows, []), (self.training_rows, []),
            )) as read,
            mock.patch.object(train, "TinyGroundedQA", return_value=model),
            mock.patch.object(train.torch.optim, "Adam") as optimizer,
            mock.patch.object(train, "measure_loss", side_effect=(2.0, 1.0)) as measure,
            mock.patch.object(train, "check_destination"),
            mock.patch.object(train, "save_checkpoint") as save,
            mock.patch.object(train, "load_checkpoint", return_value=(model, {})),
            redirect_stdout(output),
        ):
            self.assertEqual(train.main(), 0)
        self.assertEqual(read.call_args_list, [mock.call(PRACTICE_PATH), mock.call(train.TRAIN_PATH)])
        optimizer.return_value.step.assert_called_once_with()
        self.assertTrue(torch.equal(model.scores, original_scores))
        save.assert_called_once()
        path, saved_model, metadata = save.call_args.args
        self.assertEqual((path, saved_model), (destination, model))
        self.assertEqual(metadata["training_file"], PRACTICE_PATH.as_posix())
        self.assertEqual(metadata["example_count"], 20)
        self.assertEqual(metadata["example_ids"], [row["example_id"] for row in self.expanded_rows])
        self.assertEqual(metadata["source_ids"], [SOURCE])
        self.assertEqual(metadata["selection_mode"], "source_group")
        self.assertTrue(all(len(call.args[1]) == 20 for call in measure.call_args_list))
        self.assertIn(f"Training file: {PRACTICE_PATH}", output.getvalue())
        self.assertIn("Selected 20 of 20 training examples.", output.getvalue())
        self.assertIn("identical scores on all 20 practice examples", output.getvalue())


if __name__ == "__main__":
    unittest.main()
