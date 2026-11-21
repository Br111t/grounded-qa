"""Unit tests for V5K grouped joint-score/probability diagnostic."""
import pytest

from experiments.question_aware_v5i.diagnose_a_prime_field_distributions import (
    describe, grouped, quantile,
)


def test_quantile_and_distribution():
    assert quantile([4., 1., 3., 2.], 0.5) == 2.5
    assert describe([{"x": 1.}, {"x": 3.}], "x")["mean"] == 2.
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_groups_separate_field_presence_without_retraining():
    rows = []
    for field in ("leader", "location"):
        for present in (True, False):
            rows.append({
                "requested_field": field, "answerable": present,
                "max_joint": 8. if present else -4.,
                "calibrated_probability": 0.8 if present else 0.2,
                "candidate_count": 3 if present else 2,
                "top_candidate_field": field,
            })
    result = grouped(rows)
    assert result["leader_present"]["distribution"]["max_joint"]["mean"] == 8.
    assert result["location_missing"]["distribution"]["max_joint"]["mean"] == -4.
    assert result["location_present"]["false_abstentions"] == 0
    assert result["leader_missing"]["unsupported_answers"] == 0
