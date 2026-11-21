"""Tests for V5K probability threshold read-only diagnostics."""
from experiments.question_aware_v5i.diagnose_a_prime_probabilities import (
    calibrated_probability, threshold_metrics,
)


def test_calibration_probability_monotonic_joint():
    calibration = {
        "train_mean": 10., "train_std": 2.,
        "lambda": 3., "bias_delta": 0.,
    }
    low = {"max_joint": 8., "answerability_logit": 2.}
    high = {"max_joint": 12., "answerability_logit": 2.}
    pl, _ = calibrated_probability(low, calibration)
    ph, _ = calibrated_probability(high, calibration)
    assert pl < 0.5 < ph


def test_threshold_metrics_track_false_abstention_and_unsupported():
    rows = [
        {"answerable": True, "calibrated_probability": 0.4,
         "correct_top_field_if_answered": True},
        {"answerable": True, "calibrated_probability": 0.8,
         "correct_top_field_if_answered": True},
        {"answerable": False, "calibrated_probability": 0.2,
         "correct_top_field_if_answered": False},
        {"answerable": False, "calibrated_probability": 0.7,
         "correct_top_field_if_answered": False},
    ]
    score = threshold_metrics(rows, 0.5)
    assert score["correct"] == 2
    assert score["false_abstentions"] == 1
    assert score["unsupported_answers"] == 1
    lower = threshold_metrics(rows, 0.3)
    assert lower["false_abstentions"] == 0
    assert lower["unsupported_answers"] == 1
    higher = threshold_metrics(rows, 0.9)
    assert higher["false_abstentions"] == 2
    assert higher["unsupported_answers"] == 0
