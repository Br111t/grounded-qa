"""Check source selection without training a Transformer or writing checkpoints."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import predict
import train
from tokenizer import EOS_ID, VOCAB_SIZE


SOURCE = "generated_workshop_0001"
OTHER_SOURCE = "generated_workshop_0002"
HELDOUT_SOURCE = "generated_workshop_0003"
QUESTION_ORDER = (1, 2, 4, 5, 3, 10, 6, 8, 7, 9)


def make_source(source_id):
    """A small fixture with the same visible location/leader facts for every row."""
    facts = {"location": "Room Cedar", "leader": "Mira"}
    requests = (
        ("date",), ("location",), ("leader",), ("seats",),
        ("date", "location"), ("date", "leader"), ("date", "seats"),
        ("location", "leader"), ("location", "seats"), ("leader", "seats"),
    )
    rows = []
    for number, fields in enumerate(requests, 1):
        supported = [field for field in fields if field in facts]
        missing = [field for field in fields if field not in facts]
        label = "unanswerable" if not supported else "partial" if missing else "answerable"
        parts = [f"{field.title()}: {facts[field]}." for field in supported]
        parts.extend(f"{field.title()}: Insufficient information." for field in missing)
        rows.append({
            "example_id": f"{source_id}_q{number:02d}", "source_id": source_id,
            "information": f"Fictional {source_id}. Location: Room Cedar. Leader: Mira.",
            "question": "What are the " + " and ".join(fields) + "?",
            "answerability": label,
            "answer": " ".join(parts) if supported else "Insufficient information.",
        })
    return rows


class FakeModel(nn.Module):
    """Tiny differentiable scores; the optimizer and checkpoint I/O are mocked."""
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, source, decoder):
        return self.scores.view(1, 1, -1).expand(source.shape[0], decoder.shape[1], -1)


class SourceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        first, second = make_source(SOURCE), make_source(OTHER_SOURCE)
        self.rows = [row for number in QUESTION_ORDER for row in (first[number - 1], second[number - 1])]
        self.source_rows = [row for row in self.rows if row["source_id"] == SOURCE]

    def test_defaults_still_select_the_same_three_examples(self):
        expected = [f"{SOURCE}_q02", f"{SOURCE}_q05", f"{SOURCE}_q01"]
        for selector in (train.select_examples, predict.select_examples):
            with self.subTest(selector=selector.__module__):
                self.assertEqual([row["example_id"] for row in selector(self.rows)], expected)

    def test_complete_source_includes_leader_and_preserves_interleaved_file_order(self):
        expected_ids = [f"{SOURCE}_q{number:02d}" for number in QUESTION_ORDER]
        for selector in (train.select_examples, predict.select_examples):
            with self.subTest(selector=selector.__module__):
                selected = selector(self.rows, source_id=SOURCE)
                self.assertEqual([row["example_id"] for row in selected], expected_ids)
                self.assertEqual({row["source_id"] for row in selected}, {SOURCE})
                self.assertEqual(len(selected), 10)
                self.assertEqual({row["example_id"] for row in selected},
                                 {f"{SOURCE}_q{number:02d}" for number in range(1, 11)})
                leader = next(row for row in selected if row["example_id"].endswith("_q03"))
                self.assertIn("leader", leader["question"])

    def test_explicit_prediction_example_still_selects_one_row(self):
        selected = predict.select_examples(self.rows, example_id=f"{SOURCE}_q03")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["example_id"], f"{SOURCE}_q03")

    def test_unknown_sources_and_conflicting_helper_selectors_are_rejected(self):
        for selector in (train.select_examples, predict.select_examples):
            with self.subTest(selector=selector.__module__):
                with self.assertRaises(ValueError):
                    selector(self.rows, source_id=HELDOUT_SOURCE)
        with self.assertRaises(ValueError):
            predict.select_examples(self.rows, example_id=f"{SOURCE}_q03", source_id=SOURCE)

    def test_missing_or_heldout_training_source_fails_before_model_and_optimizer(self):
        for source_id in ("missing_source", HELDOUT_SOURCE):
            with self.subTest(source_id=source_id):
                arguments = ["train.py", "--source-id", source_id, "--steps", "1"]
                with (
                    mock.patch.object(sys, "argv", arguments),
                    mock.patch.object(train, "validate_data", return_value=(self.rows, [])) as read,
                    mock.patch.object(train, "TinyGroundedQA") as model,
                    mock.patch.object(train.torch.optim, "Adam") as optimizer,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(train.main(), 1)
                read.assert_called_once_with(train.TRAIN_PATH)
                model.assert_not_called()
                optimizer.assert_not_called()

    def test_prediction_cli_rejects_conflicting_options_before_io(self):
        conflicts = [
            ["--source-id", SOURCE, "--example-id", f"{SOURCE}_q03"],
            *[["--information", "Location: Cedar.", "--question", "Where?", flag, value]
              for flag, value in (("--source-id", SOURCE), ("--example-id", "q03"), ("--data", "unused.jsonl"))],
        ]
        for options in conflicts:
            with self.subTest(options=options):
                with (
                    mock.patch.object(sys, "argv", ["predict.py", "--checkpoint", "unused.pt"] + options),
                    mock.patch.object(predict, "validate_data") as read,
                    mock.patch.object(predict, "load_checkpoint") as load,
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        predict.main()
                    self.assertEqual(error.exception.code, 2)
                read.assert_not_called()
                load.assert_not_called()

    def test_prediction_cli_generates_every_selected_question_in_order(self):
        model = object()
        result = predict.GenerationResult("", [EOS_ID], True, True)
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["predict.py", "--checkpoint", "unused.pt", "--source-id", SOURCE]),
            mock.patch.object(predict, "validate_data", return_value=(self.rows, [])),
            mock.patch.object(predict, "load_checkpoint", return_value=(model, {})),
            mock.patch.object(predict, "generate_answer", return_value=result) as generate,
            redirect_stdout(output),
        ):
            self.assertEqual(predict.main(), 0)
        self.assertEqual(generate.call_args_list, [
            mock.call(model, row["information"], row["question"], max_new_tokens=None)
            for row in self.source_rows
        ])
        self.assertIn("Previewing 10 examples", output.getvalue())

    def test_training_checkpoint_metadata_and_output_use_actual_selection_count(self):
        for source_id, count, mode in ((None, 3, "first_per_type"), (SOURCE, 10, "source_group")):
            with self.subTest(selection_mode=mode):
                model = FakeModel()
                original = model.scores.detach().clone()
                destination = Path("unused-test-checkpoint.pt")
                arguments = ["train.py", "--steps", "1", "--checkpoint", str(destination)]
                if source_id is not None:
                    arguments.extend(["--source-id", source_id])
                output = io.StringIO()
                with (
                    mock.patch.object(sys, "argv", arguments),
                    mock.patch.object(train, "validate_data", return_value=(self.rows, [])) as read,
                    mock.patch.object(train, "TinyGroundedQA", return_value=model),
                    mock.patch.object(train.torch.optim, "Adam") as optimizer,
                    mock.patch.object(train, "measure_loss", side_effect=(2.0, 1.0)) as measure,
                    mock.patch.object(train, "check_destination"),
                    mock.patch.object(train, "save_checkpoint") as save,
                    mock.patch.object(train, "load_checkpoint", return_value=(model, {})),
                    redirect_stdout(output),
                ):
                    self.assertEqual(train.main(), 0)
                read.assert_called_once_with(train.TRAIN_PATH)
                optimizer.return_value.step.assert_called_once_with()
                self.assertTrue(torch.equal(model.scores, original))
                save.assert_called_once()
                path, saved_model, metadata = save.call_args.args
                self.assertEqual((path, saved_model), (destination, model))
                expected = train.select_examples(self.rows, source_id=source_id)
                self.assertEqual(metadata["example_ids"], [row["example_id"] for row in expected])
                self.assertEqual(metadata["source_ids"], [SOURCE])
                self.assertEqual(metadata["example_count"], count)
                self.assertEqual(metadata["selection_mode"], mode)
                self.assertEqual(metadata["steps"], 1)
                self.assertTrue(all(len(call.args[1]) == count for call in measure.call_args_list))
                self.assertIn(f"Selected {count} of 20 training examples.", output.getvalue())
                self.assertIn(f"these same {count} training examples", output.getvalue())
                self.assertIn(f"identical scores on all {count} practice examples", output.getvalue())
                self.assertIn("Complete passes through selected examples: 0; additional updates: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
