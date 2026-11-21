"""Tests for V5J read-only score-separation statistics."""
import pytest

from experiments.question_aware_v5i.diagnose_bilinear_answerability import (
    rank_auc, summary,
)


def test_auc_perfect_separation_and_reversal():
    assert rank_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert rank_auc([0.0, 1.0], [2.0, 3.0]) == 0.0


def test_auc_ties():
    assert rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        rank_auc([], [1.0])


def test_summary_preserves_present_missing_class_mean_and_auc():
    rows = [
        {"answerable": True, "max_end": 2.0},
        {"answerable": True, "max_end": 4.0},
        {"answerable": False, "max_end": 0.0},
        {"answerable": False, "max_end": 1.0},
    ]
    result = summary(rows, "max_end")
    assert result["answerable_mean"] == 3.0
    assert result["missing_mean"] == 0.5
    assert result["auc_high_implies_answerable"] == 1.0
