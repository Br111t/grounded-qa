"""Tests for unchanged-weight decoder and paired outcome accounting."""
import unittest

import torch

from experiments.evidence_sensitivity_v3.generate_evidence_data_v3 import information_for
from experiments.field_boundary_decoder_v5e.field_decoder import (
    complete_field_spans, predict_complete_field,
)
from experiments.field_boundary_decoder_v5e.compare_decoders import aggregate, summarize, verify_baseline
from experiments.field_boundary_decoder_v5e.generate_holdout import build_holdout
from grounded_qa.models.span_model_v5 import prepare_span_input
from grounded_qa.tokenization.tokenizer import encode


class FakeConfig:
    max_span_bytes = 16


class FakeModel(torch.nn.Module):
    def __init__(self, starts, ends, answer_logit=8):
        super().__init__()
        self.config = FakeConfig()
        self.starts = starts
        self.ends = ends
        self.answer_logit = answer_logit

    def forward(self, tokens, mask):
        start = torch.full(tokens.shape, -10.0)
        end = torch.full(tokens.shape, -10.0)
        for p, v in self.starts.items():
            start[0, p] = v
        for p, v in self.ends.items():
            end[0, p] = v
        return start, end, torch.tensor([float(self.answer_logit)])


class FieldDecoderTests(unittest.TestCase):
    def setUp(self):
        self.information = information_for("Dblt", ("Room Pine", False, "Python"))
        self.question = "What is the leader of Workshop W0001?"

    def test_all_fields_and_offsets_are_source_grounded(self):
        candidates = complete_field_spans(self.information)
        self.assertEqual({c["field"] for c in candidates},
                         {"Leader", "Location", "Topic"})
        tokens, mask = prepare_span_input(self.information, self.question)
        for c in candidates:
            self.assertTrue(mask[0, c["start"]:c["end"] + 1].all().item())
            self.assertEqual(tokens.shape[1] > c["end"], True)
            self.assertEqual(c["end"] - c["start"] + 1,
                             len(encode(c["value"])))

    def test_candidate_decoder_chooses_by_scores_not_gold_or_field_name(self):
        candidates = complete_field_spans(self.information)
        indexed = {c["field"]: c for c in candidates}
        model = FakeModel(
            {indexed["Leader"]["start"]: 5, indexed["Topic"]["start"]: 8},
            {indexed["Leader"]["end"]: 5, indexed["Topic"]["end"]: 8},
        )
        result, probability, selected, scores = predict_complete_field(
            model, self.information, self.question)
        self.assertEqual(result, "Leader: Python.")
        self.assertEqual(selected,
                         (indexed["Topic"]["start"], indexed["Topic"]["end"]))
        self.assertEqual(scores[0]["field"], "Topic")
        self.assertGreater(probability, 0.5)

    def test_no_answer_keeps_original_gate(self):
        model = FakeModel({}, {}, answer_logit=-10)
        result, _, selected, scores = predict_complete_field(
            model, self.information, self.question)
        self.assertEqual(result, "Insufficient information.")
        self.assertIsNone(selected)
        self.assertEqual(scores, [])

    def test_improvement_regression_and_unchanged_are_disjoint(self):
        rows = [
            {"baseline_correct": False, "alternative_correct": True,
             "status": "improved", "changed_prediction": True},
            {"baseline_correct": True, "alternative_correct": False,
             "status": "regressed", "changed_prediction": True},
            {"baseline_correct": False, "alternative_correct": False,
             "status": "unchanged", "changed_prediction": True},
            {"baseline_correct": True, "alternative_correct": True,
             "status": "unchanged", "changed_prediction": False},
        ]
        result = aggregate(rows)
        self.assertEqual((result["improved"], result["regressed"],
                          result["unchanged"], result["changed_predictions"]),
                         (1, 1, 2, 3))
        self.assertEqual(result["baseline_exact"], result["alternative_exact"])

    def test_baseline_drift_refused(self):
        sample = {"example_id": "x"}
        frozen = {("v5", "x"): {
            "predicted": "Leader: Dblt.", "selected_start": 2,
            "selected_end": 5,
        }}
        verify_baseline(frozen, "v5", sample, "Leader: Dblt.", (2, 5))
        with self.assertRaises(ValueError):
            verify_baseline(frozen, "v5", sample, "Leader: Dblt.", (3, 5))

    def test_new_holdout_is_108_unique_and_disjoint_name_family(self):
        rows = build_holdout()
        self.assertEqual(len(rows), 108)
        self.assertEqual(len({r["information"] for r in rows}), 108)
        answerable = [r for r in rows if r["answerability"] == "answerable"]
        self.assertEqual(len(answerable), 96)
        self.assertTrue(all(r["answer"].startswith("Leader: Eb") for r in answerable))
        self.assertEqual(sum(r["answerability"] == "unanswerable"
                             for r in rows), 12)


if __name__ == "__main__":
    unittest.main()
