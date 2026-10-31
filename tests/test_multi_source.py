"""Check multi-source selection without training a Transformer or saving files."""

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

import grounded_qa.inference.predict as predict
import grounded_qa.training.train as train
from grounded_qa.tokenization.tokenizer import EOS_ID, VOCAB_SIZE


SOURCE = "generated_workshop_0001"
OTHER_SOURCE = "generated_workshop_0002"
MISSING_SOURCE = "generated_workshop_0003"
QUESTION_ORDER = (1, 2, 4, 5, 3, 10, 6, 8, 7, 9)
PRACTICE_PATH = Path("data/training/unused-multi-source-practice.jsonl")


def make_source(source_id, facts):
    """Ten questions whose targets use only this source's supplied facts."""
    requests = (
        ("date",), ("location",), ("leader",), ("seats",),
        ("date", "location"), ("date", "leader"), ("date", "seats"),
        ("location", "leader"), ("location", "seats"), ("leader", "seats"),
    )
    information = f"Fictional {source_id}. " + " ".join(
        f"{field.title()}: {value}." for field, value in facts.items()
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
            "information": information,
            "question": "What are the " + " and ".join(fields) + "?",
            "answerability": label,
            "answer": " ".join(parts) if supported else "Insufficient information.",
        })
    return rows


class FakeModel(nn.Module):
    """Differentiable scores for CLI wiring; the mocked optimizer cannot update them."""

    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, source, decoder):
        return self.scores.view(1, 1, -1).expand(source.shape[0], decoder.shape[1], -1)


class MultiSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        first = make_source(SOURCE, {"location": "Room Cedar", "leader": "Mira"})
        second = make_source(OTHER_SOURCE, {"date": "2031-04-21", "seats": "18"})
        self.rows = [
            row for number in QUESTION_ORDER
            for row in (first[number - 1], second[number - 1])
        ]

    def test_defaults_and_single_source_keep_existing_selection(self):
        default_ids = [f"{OTHER_SOURCE}_q01", f"{SOURCE}_q05", f"{SOURCE}_q01"]
        source_rows = [row for row in self.rows if row["source_id"] == SOURCE]
        for selector in (train.select_examples, predict.select_examples):
            with self.subTest(selector=selector.__module__):
                defaults = selector(self.rows)
                self.assertEqual([row["example_id"] for row in defaults], default_ids)
                self.assertEqual(selector(self.rows, source_ids=None), defaults)
                self.assertEqual(selector(self.rows, source_id=SOURCE), source_rows)
                self.assertEqual(selector(self.rows, source_ids=[SOURCE]), source_rows)
                self.assertEqual(len(source_rows), 10)

    def test_multiple_sources_preserve_file_order_and_each_sources_facts(self):
        # Reversing argument order must not regroup the interleaved file rows.
        unrequested = make_source(MISSING_SOURCE, {"location": "Room Birch", "seats": "20"})
        for selector in (train.select_examples, predict.select_examples):
            with self.subTest(selector=selector.__module__):
                selected = selector(self.rows + unrequested, source_ids=[OTHER_SOURCE, SOURCE])
                self.assertEqual(selected, self.rows)
                self.assertEqual(len(selected), 20)
                self.assertEqual(len({row["information"] for row in selected}), 2)
                for actual, original in zip(selected, self.rows):
                    self.assertIs(actual, original)

    def test_empty_duplicate_or_missing_source_ids_are_rejected(self):
        invalid_lists = (
            [], [""], [SOURCE, " "], [SOURCE, SOURCE],
            [MISSING_SOURCE], [SOURCE, MISSING_SOURCE],
        )
        for selector in (train.select_examples, predict.select_examples):
            for source_ids in invalid_lists:
                with self.subTest(selector=selector.__module__, source_ids=source_ids):
                    with self.assertRaises(ValueError):
                        selector(self.rows, source_ids=source_ids)

    def test_helpers_reject_conflicting_selectors(self):
        for selector in (train.select_examples, predict.select_examples):
            with self.subTest(selector=selector.__module__):
                with self.assertRaises(ValueError):
                    selector(self.rows, source_id=SOURCE, source_ids=[SOURCE, OTHER_SOURCE])
        with self.assertRaises(ValueError):
            predict.select_examples(
                self.rows, example_id=f"{SOURCE}_q03", source_ids=[SOURCE, OTHER_SOURCE],
            )

    def test_training_cli_conflicts_and_empty_list_fail_before_io(self):
        conflicts = (
            ["--source-id", SOURCE, "--source-ids", SOURCE, OTHER_SOURCE],
            ["--source-ids", SOURCE, OTHER_SOURCE, "--source-id", SOURCE],
            ["--source-ids"],
        )
        for options in conflicts:
            with self.subTest(options=options):
                with (
                    mock.patch.object(sys, "argv", ["train.py", "--steps", "1"] + options),
                    mock.patch.object(train, "validate_data") as read,
                    mock.patch.object(train, "check_destination") as destination,
                    mock.patch.object(train, "TinyGroundedQA") as model,
                    mock.patch.object(train.torch.optim, "Adam") as optimizer,
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        train.main()
                    self.assertEqual(error.exception.code, 2)
                read.assert_not_called()
                destination.assert_not_called()
                model.assert_not_called()
                optimizer.assert_not_called()

    def test_prediction_cli_conflicts_and_empty_list_fail_before_io(self):
        conflicts = (
            ["--source-id", SOURCE, "--source-ids", SOURCE, OTHER_SOURCE],
            ["--example-id", f"{SOURCE}_q03", "--source-ids", SOURCE, OTHER_SOURCE],
            ["--information", "Location: Cedar.", "--question", "Where?",
             "--source-ids", SOURCE, OTHER_SOURCE],
            ["--source-ids"],
        )
        for options in conflicts:
            with self.subTest(options=options):
                with (
                    mock.patch.object(sys, "argv", ["predict.py", "--checkpoint", "unused.pt"] + options),
                    mock.patch.object(predict, "validate_data") as read,
                    mock.patch.object(predict, "load_checkpoint") as load,
                    mock.patch.object(predict, "generate_answer") as generate,
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as error:
                        predict.main()
                    self.assertEqual(error.exception.code, 2)
                read.assert_not_called()
                load.assert_not_called()
                generate.assert_not_called()

    def test_training_invalid_list_stops_before_model_optimizer_or_save(self):
        for source_ids in ([SOURCE, MISSING_SOURCE], [SOURCE, SOURCE]):
            with self.subTest(source_ids=source_ids):
                with (
                    mock.patch.object(sys, "argv", ["train.py", "--steps", "1", "--source-ids"] + source_ids),
                    mock.patch.object(train, "validate_data", return_value=(self.rows, [])),
                    mock.patch.object(train, "TinyGroundedQA") as model,
                    mock.patch.object(train.torch.optim, "Adam") as optimizer,
                    mock.patch.object(train, "save_checkpoint") as save,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(train.main(), 1)
                model.assert_not_called()
                optimizer.assert_not_called()
                save.assert_not_called()

    def test_prediction_invalid_list_stops_before_checkpoint_or_generation(self):
        for source_ids in ([SOURCE, MISSING_SOURCE], [SOURCE, SOURCE]):
            with self.subTest(source_ids=source_ids):
                with (
                    mock.patch.object(sys, "argv", [
                        "predict.py", "--checkpoint", "unused.pt", "--source-ids",
                    ] + source_ids),
                    mock.patch.object(predict, "validate_data", return_value=(self.rows, [])),
                    mock.patch.object(predict, "load_checkpoint") as load,
                    mock.patch.object(predict, "generate_answer") as generate,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(predict.main(), 1)
                load.assert_not_called()
                generate.assert_not_called()

    def test_prediction_cli_generates_both_sources_in_file_order(self):
        model = object()
        result = predict.GenerationResult("", [EOS_ID], True, True)
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [
                "predict.py", "--checkpoint", "unused.pt", "--data", str(PRACTICE_PATH),
                "--source-ids", OTHER_SOURCE, SOURCE,
            ]),
            mock.patch.object(predict, "validate_data", return_value=(self.rows, [])) as read,
            mock.patch.object(predict, "load_checkpoint", return_value=(model, {})),
            mock.patch.object(predict, "generate_answer", return_value=result) as generate,
            redirect_stdout(output),
        ):
            self.assertEqual(predict.main(), 0)
        read.assert_called_once_with(PRACTICE_PATH)
        self.assertEqual(generate.call_args_list, [
            mock.call(model, row["information"], row["question"], max_new_tokens=None, prompt_version="original")
            for row in self.rows
        ])
        self.assertIn(f"Previewing 20 examples from {PRACTICE_PATH}.", output.getvalue())
        self.assertIn("Source groups: 2", output.getvalue())

    def test_training_cli_records_actual_path_counts_and_ordered_ids(self):
        model = FakeModel()
        original_scores = model.scores.detach().clone()
        destination = Path("unused-multi-source-test-checkpoint.pt")
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [
                "train.py", "--data", str(PRACTICE_PATH), "--steps", "1",
                "--checkpoint", str(destination), "--source-ids", OTHER_SOURCE, SOURCE,
            ]),
            mock.patch.object(train, "validate_data", side_effect=(
                (self.rows, []), (self.rows, []),
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
        self.assertEqual(metadata["source_ids"], [SOURCE, OTHER_SOURCE])
        self.assertEqual(metadata["source_count"], 2)
        self.assertEqual(metadata["example_ids"], [row["example_id"] for row in self.rows])
        self.assertEqual(metadata["example_count"], 20)
        self.assertEqual(metadata["selection_mode"], "source_groups")
        self.assertTrue(all(len(call.args[1]) == 20 for call in measure.call_args_list))
        self.assertIn(f"Training file: {PRACTICE_PATH}", output.getvalue())
        self.assertIn("Selected 20 of 20 training examples.", output.getvalue())
        self.assertIn("Source groups: 2", output.getvalue())
        self.assertIn("identical scores on all 20 practice examples", output.getvalue())


if __name__ == "__main__":
    unittest.main()
