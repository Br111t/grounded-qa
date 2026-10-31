"""Tests for the read-only V5C matched length × ending diagnostic."""

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.evidence_sensitivity_v3.evidence_v3_validation import parse_fields
from experiments.span_boundary_probe_v5c.evaluate_span_probe_v5c import _check_checkpoint, evaluate_probe
from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import build_datasets
from experiments.variable_span_v5b.generate_span_data_v5b import build_additions
from experiments.span_boundary_probe_v5c.generate_span_probe_v5c import (
    GROUPS, ROWS_PER_GROUP, build_probe_datasets,
    validate_probe_group,
)
from grounded_qa.models.span_model_v5 import SpanGroundedQA, prepare_span_input, training_span_target
from grounded_qa.tokenization.tokenizer import encode


def names(groups):
    return {
        parse_fields(row["information"], row["source_id"])["Leader"]
        for records in groups.values() for row in records
        if row["answerability"] == "answerable"
    }


class SpanBoundaryProbeV5CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.probes = build_probe_datasets()

    def test_locked_reproducibility_and_factor_balance(self):
        self.assertEqual(self.probes, build_probe_datasets())
        self.assertEqual(set(self.probes), set(GROUPS))
        for group in GROUPS:
            records = self.probes[group]
            self.assertEqual(len(records), ROWS_PER_GROUP)
            validate_probe_group(group, records)
            leaders = [
                parse_fields(row["information"], row["source_id"])["Leader"]
                for row in records
            ]
            self.assertEqual(len(set(leaders)), 12)
            self.assertEqual(set(Counter(leaders).values()), {2})
            expected_length = 4 if "length4" in group else 5
            self.assertTrue(all(len(leader) == expected_length for leader in leaders))
            expected_vowel = group.endswith("vowel")
            self.assertTrue(all((leader[-1] in "aeiou") == expected_vowel for leader in leaders))
            last_counts = Counter(leader[-1] for leader in leaders)
            # Both terminal letters occur equally often within each category.
            self.assertEqual(sorted(last_counts.values()), [12, 12])

    def test_matched_contexts_and_name_stems_across_all_groups(self):
        for index in range(ROWS_PER_GROUP):
            data = {
                group: parse_fields(self.probes[group][index]["information"], "probe")
                for group in GROUPS
            }
            contexts = {
                (fields["Location"], fields.get("Topic"))
                for fields in data.values()
            }
            self.assertEqual(len(contexts), 1)
            a = data["length4_vowel"]["Leader"]
            b = data["length4_consonant"]["Leader"]
            c = data["length5_vowel"]["Leader"]
            d = data["length5_consonant"]["Leader"]
            self.assertEqual(a[:3], b[:3])
            self.assertEqual(a[:3], c[:3])
            self.assertEqual(a[:3], d[:3])
            self.assertEqual(c[3], "e")
            self.assertEqual(d[3], "e")
            self.assertEqual(a[-1], c[-1])
            self.assertEqual(b[-1], d[-1])
            # Same field ordering across the paired examples.
            leader_first = [
                row["information"].index("Leader: ") < row["information"].index("Location: ")
                for group in GROUPS for row in (self.probes[group][index],)
            ]
            self.assertEqual(len(set(leader_first)), 1)

    def test_no_source_name_or_evidence_leakage(self):
        old = {**build_datasets(), **build_additions()}
        self.assertTrue(names(self.probes).isdisjoint(names(old)))
        self.assertEqual(len(names(self.probes)), 48)
        all_rows = [r for group in (*old.values(), *self.probes.values()) for r in group]
        self.assertEqual(len(all_rows), len({r["source_id"] for r in all_rows}))
        self.assertEqual(len(all_rows), len({r["information"] for r in all_rows}))

    def test_probe_labels_match_real_source_bytes(self):
        for group in GROUPS:
            for row in self.probes[group]:
                tokens, mask = prepare_span_input(row["information"], row["question"])
                answerable, start, end = training_span_target(row, tokens, mask)
                self.assertEqual(answerable, 1.0)
                self.assertTrue(mask[0, start:end + 1].all().item())
                self.assertEqual(
                    tokens[0, start:end + 1].tolist(),
                    encode(row["answer"][len("Leader: "):-1]),
                )

    def test_corrupted_probe_reference_rejected(self):
        bad = copy.deepcopy(self.probes["length4_vowel"])
        bad[0]["answer"] = "Leader: NotTheAnswer."
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_probe_group("length4_vowel", bad)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_probe_group("training", bad)

    def test_evaluation_does_not_change_weights(self):
        torch.manual_seed(42)
        model = SpanGroundedQA().eval()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        # Use one matched row per group for a quick diagnostic smoke test.
        sample = {g: rows[:1] for g, rows in self.probes.items()}
        from contextlib import redirect_stdout
        from io import StringIO
        with redirect_stdout(StringIO()):
            scores = evaluate_probe(model, sample, show_errors=0)
        self.assertEqual(set(scores), set(GROUPS))
        self.assertTrue(all(score["total"] == 1 for score in scores.values()))
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], tensor))

    def test_checkpoint_registration_guard(self):
        from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import OUTPUTS
        info = {
            "training_file": OUTPUTS["train"].as_posix(),
            "answerability_threshold": 0.5,
        }
        _check_checkpoint(info)
        with self.assertRaisesRegex(ValueError, "registered V3"):
            _check_checkpoint({**info, "training_file": "data/evaluation/not_train.jsonl"})
        with self.assertRaisesRegex(ValueError, "threshold"):
            _check_checkpoint({**info, "answerability_threshold": 0.9})


if __name__ == "__main__":
    unittest.main()
