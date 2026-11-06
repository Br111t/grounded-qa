"""V5H loss-contract tests; CPU only, no checkpoint or reserved EVAL access."""
import collections
import unittest

import torch

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import build_partition
from experiments.question_aware_v5g.train_uniform_sampling import uniform_schedule
from experiments.question_aware_v5h.train_paired_ranking import (
    paired_rank_penalty, train_loss,
)
from grounded_qa.models.span_model_v5 import span_training_loss


CANDIDATES = [
    {"field": "Leader", "start": 1, "end": 1, "value": "Fa"},
    {"field": "Location", "start": 3, "end": 3, "value": "Room"},
]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.starts = torch.nn.Parameter(torch.tensor([[0.0, 0.2, 0.0, 2.0, 0.0]]))
        self.ends = torch.nn.Parameter(torch.tensor([[0.0, 0.2, 0.0, 2.0, 0.0]]))
        self.answers = torch.nn.Parameter(torch.tensor([0.3]))

    def forward(self, ids, mask):
        return self.starts, self.ends, self.answers


class V5HPairedRankingTests(unittest.TestCase):
    def test_hinge_favors_requested_field_and_has_gradient(self):
        starts = torch.tensor([[0., 0., 0., 2., 0.]], requires_grad=True)
        ends = torch.tensor([[0., 0., 0., 2., 0.]], requires_grad=True)
        penalty, eligible = paired_rank_penalty(
            starts, ends, CANDIDATES, "leader", 1.0
        )
        self.assertTrue(eligible)
        self.assertAlmostEqual(penalty.item(), 5.0)
        penalty.backward()
        self.assertLess(starts.grad[0, 1].item(), 0.0)
        self.assertGreater(starts.grad[0, 3].item(), 0.0)
        self.assertLess(ends.grad[0, 1].item(), 0.0)
        self.assertGreater(ends.grad[0, 3].item(), 0.0)

    def test_correct_field_with_sufficient_margin_has_no_penalty(self):
        starts = torch.tensor([[0., 0., 0., 2., 0.]], requires_grad=True)
        ends = starts.detach().clone().requires_grad_(True)
        penalty, eligible = paired_rank_penalty(
            starts, ends, CANDIDATES, "location"
        )
        self.assertTrue(eligible)
        self.assertEqual(penalty.item(), 0.0)

    def test_missing_rival_has_zero_ranking_gradient(self):
        starts = torch.randn(1, 5, requires_grad=True)
        ends = torch.randn(1, 5, requires_grad=True)
        penalty, eligible = paired_rank_penalty(
            starts, ends, CANDIDATES[:1], "leader"
        )
        self.assertFalse(eligible)
        self.assertEqual(penalty.item(), 0.0)
        penalty.backward()
        self.assertTrue(torch.equal(starts.grad, torch.zeros_like(starts)))
        self.assertIsNone(ends.grad)

    def test_base_objective_is_unchanged_when_rank_weight_zero(self):
        model = TinyModel()
        ids = torch.ones((1, 5), dtype=torch.long)
        mask = torch.ones((1, 5), dtype=torch.bool)
        for target in ((1.0, 1, 1), (0.0, None, None)):
            original, _, _ = span_training_loss(model, ids, mask, target)
            revised, _, eligible = train_loss(
                model, ids, mask, target, CANDIDATES, "leader",
                rank_weight=0.0,
            )
            self.assertAlmostEqual(original.item(), revised.item(), places=6)
            self.assertEqual(eligible, bool(target[0]))

    def test_rank_does_not_apply_to_unanswerable_example(self):
        model = TinyModel()
        ids = torch.ones((1, 5), dtype=torch.long)
        mask = torch.ones((1, 5), dtype=torch.bool)
        total, rank, eligible = train_loss(
            model, ids, mask, (0.0, None, None), CANDIDATES, "leader",
        )
        self.assertFalse(eligible)
        self.assertEqual(rank.item(), 0.0)
        self.assertTrue(torch.isfinite(total).item())

    def test_registered_data_and_schedule_leave_holdout_sealed(self):
        rows = build_partition("train")
        self.assertEqual(len(rows), 48)
        eligible_rows = [
            r for r in rows
            if r["answerability"] == "answerable" and
            {"leader", "location"}.issubset({
                c["field"].lower() for c in complete_field_spans(r["information"])
            })
        ]
        self.assertEqual(len(eligible_rows), 24)
        schedule = uniform_schedule(48, 1200, 42)
        counts = collections.Counter(schedule)
        self.assertEqual(set(counts.values()), {25})
        self.assertEqual(
            sum(rows[index]["example_id"] in
                {r["example_id"] for r in eligible_rows} for index in schedule),
            600,
        )


if __name__ == "__main__":
    unittest.main()
