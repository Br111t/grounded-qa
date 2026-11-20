"""V5I interface, gradients and checkpoint-isolation tests (no dataset access)."""
import pytest
import torch

from grounded_qa.models.span_model_v5 import SpanGroundedQA, prepare_span_input
from grounded_qa.models.span_model_v5i import (
    QuestionConditionedSpanQA,
    load_v5i_checkpoint,
    save_v5i_checkpoint,
)


def sample(question="Who is the leader?"):
    return prepare_span_input(
        "Fictional Workshop W0001. Leader: Fbat. Location: Birch 25. Topic: Python.",
        question,
    )


def test_question_mask_excludes_evidence_and_framing():
    ids, evidence = sample()
    question = QuestionConditionedSpanQA.question_mask(ids, evidence)
    assert question.dtype == torch.bool
    assert not (question & evidence).any().item()
    assert question.sum().item() == len("Who is the leader?".encode("utf-8"))


def test_different_questions_have_distinct_masks_and_valid_logits():
    model = QuestionConditionedSpanQA().eval()
    leader_ids, leader_evidence = sample("Who is the leader?")
    location_ids, location_evidence = sample("Where is this workshop located?")
    outputs = []
    for ids, evidence in [(leader_ids, leader_evidence), (location_ids, location_evidence)]:
        start, end, answer = model(ids, evidence)
        outputs.append((start, end, answer))
        assert start.shape == end.shape == ids.shape
        assert answer.shape == (1,)
        assert torch.isfinite(start[evidence]).all().item()
        assert torch.isfinite(end[evidence]).all().item()
        assert torch.isfinite(answer).all().item()
        assert (start[~evidence] == -1e9).all().item()
    # Fixed evidence, different questions: output must be question-sensitive
    # at the initialized readout, even before learning useful field selection.
    first_start, _, _ = outputs[0]
    second_start, _, _ = outputs[1]
    # Compare evidence-only slices: question length changes the total tensor length.
    first_values = first_start[leader_evidence]
    second_values = second_start[location_evidence]
    assert first_values.shape == second_values.shape
    assert not torch.allclose(first_values, second_values)


def test_fusion_receives_training_gradient():
    torch.manual_seed(42)
    model = QuestionConditionedSpanQA()
    ids, evidence = sample()
    start, end, answer = model(ids, evidence)
    loss = start[evidence].square().mean() + end[evidence].square().mean() + answer.square().mean()
    loss.backward()
    grad = model.question_fusion.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all().item()
    assert grad.abs().sum().item() > 0
    assert sum(p.numel() for p in model.parameters()) > sum(
        p.numel() for p in SpanGroundedQA().parameters()
    )


def test_checkpoint_roundtrip_and_baseline_isolation(tmp_path):
    model = QuestionConditionedSpanQA().eval()
    path = tmp_path / "v5i.pt"
    info = {"steps": 1, "experiment_variant": "v5i_question_conditioned"}
    save_v5i_checkpoint(path, model, info)
    restored, metadata = load_v5i_checkpoint(path)
    assert metadata == info
    assert set(restored.state_dict()) == set(model.state_dict())
    for key in model.state_dict():
        assert torch.equal(restored.state_dict()[key], model.state_dict()[key])
    with pytest.raises(ValueError):
        save_v5i_checkpoint(path, model, info)
    from grounded_qa.models.span_model_v5 import load_span_checkpoint
    with pytest.raises(ValueError, match="Not a supported V5"):
        load_span_checkpoint(path)


def test_malformed_separator_is_rejected():
    ids, evidence = sample()
    changed = ids.clone()
    boundary = int(torch.nonzero(evidence[0]).flatten()[-1].item()) + 1
    changed[0, boundary] = changed[0, boundary] + 1
    with pytest.raises(ValueError, match="separator"):
        QuestionConditionedSpanQA.question_mask(changed, evidence)
