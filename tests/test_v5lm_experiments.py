"""Smoke/contract tests for independent V5L/V5M experiments."""
import torch

from experiments.question_aware_v5i.diagnose_v5l_paired_questions import pair_sources
from experiments.question_aware_v5i.train_v5m_learned_null import fit_null, evaluate


def test_v5l_pairs_require_both_questions_and_detect_presence():
    records = []
    for i in range(24):
        for field in ("leader", "location"):
            records.append({
                "source_id": f"source_{i}", "requested_field": field,
                "question": f"Question: {field}?",
                "answerable": (i % 4 != (0 if field == "leader" else 1)),
                "max_joint": float(i + (field == "location")),
                "calibrated_probability": 0.7,
                "top_candidate_field": field,
            })
    paired = pair_sources(records)
    assert len(paired) == 24
    assert paired[0]["joint_delta_location_minus_leader"] == 1.


def test_v5m_null_learns_from_train_only_feature_vectors():
    # Synthetic samples test learning and exact 129-param architecture,
    # not performance or transfer on real DEV data.
    records = []
    for i in range(48):
        x = torch.zeros(128, dtype=torch.float64)
        x[0] = -2. if i < 12 else 2.
        records.append({
            "features": x,
            "max_joint": 0.0,
            "answerable": i >= 12,
        })
    model, mean, std, trajectory = fit_null(records, steps=120, lr=0.05)
    assert sum(p.numel() for p in model.parameters()) == 129
    assert trajectory[-1]["train_bce"] < trajectory[0]["train_bce"]
    assert torch.isfinite(mean).all() and (std > 0).all()
