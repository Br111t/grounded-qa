"""V5K joint-score residual calibration tests (no checkpoint required)."""
import torch

from experiments.question_aware_v5i.train_joint_a_prime import (
    classify, fit_calibrator, standardize,
)


def test_standardization_is_train_only():
    values = torch.tensor([1., 3.], dtype=torch.float64)
    assert torch.allclose(standardize(values, 2., 1.), torch.tensor([-1., 1.], dtype=torch.float64))


def test_calibrator_initially_preserves_original_logits():
    records = [
        {
            "max_joint": float(i),
            "answerability_logit": 2.0,
            "answerable": i >= 12,
        }
        for i in range(48)
    ]
    result = fit_calibrator(records, steps=100, lr=0.05)
    assert result["train_std"] > 0
    assert result["train_bce"][-1]["train_bce"] < result["train_bce"][0]["train_bce"]
    assert result["lambda"] > 0


def test_classify_abstains_below_threshold_and_keeps_evidence_value():
    rows = [{
        "example_id": "example_a", "source_id": "s1",
        "information": "Fictional Workshop W0001. Leader: Fara. Location: Cedar 12. Topic: Python.",
        "question": "What is the leader of Workshop W0001?",
        "target_field": "leader", "question_variant": "canonical",
        "expected_value": "Fara",
    }]
    records = {"example_a": {
        "example_id": "example_a", "answerability_logit": 2.0,
        "max_joint": 5.0,
        "candidates": [{"field": "leader", "joint_score": 12.0}],
    }}
    yes = classify(rows, records)[0]
    assert yes["predicted_field"] == "leader"
    assert yes["predicted_value"] == "Fara"
    no = classify(rows, records, {
        "lambda": 0.0, "bias_delta": -10.0,
        "train_mean": 0.0, "train_std": 1.0,
    })[0]
    assert no["predicted_field"] is None
    assert no["correct"] is False
