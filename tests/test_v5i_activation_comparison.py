"""Unit tests for the controlled V5I fusion activation ablation."""
import torch
from torch.nn import functional as F

from experiments.question_aware_v5i.compare_activations import ActivationSpanQA
from grounded_qa.models.span_model_v5 import prepare_span_input
from grounded_qa.models.span_model_v5i import QuestionConditionedSpanQA


def test_tanh_variant_matches_original_v5i():
    torch.manual_seed(42)
    original = QuestionConditionedSpanQA().eval()
    torch.manual_seed(42)
    variant = ActivationSpanQA("tanh").eval()
    original_state = original.state_dict()
    variant_state = variant.state_dict()
    assert set(original_state) == set(variant_state)
    assert all(torch.equal(original_state[k], variant_state[k]) for k in original_state)
    ids, mask = prepare_span_input(
        "Leader: Fara. Location: Reno. Topic: Training.",
        "Who is the leader?",
    )
    with torch.no_grad():
        left = original(ids, mask)
        right = variant(ids, mask)
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_all_variants_have_identical_initial_parameters_and_shapes():
    models = []
    for activation in ("tanh", "gelu", "linear"):
        torch.manual_seed(42)
        model = ActivationSpanQA(activation).eval()
        models.append(model)
    for key, weight in models[0].state_dict().items():
        assert all(torch.equal(weight, m.state_dict()[key]) for m in models[1:])
    ids, mask = prepare_span_input(
        "Leader: Fara. Location: Reno. Topic: Training.",
        "Where is the location?",
    )
    with torch.no_grad():
        outputs = [model(ids, mask) for model in models]
    for starts, ends, answer in outputs:
        assert starts.shape == ends.shape == ids.shape
        assert answer.shape == (1,)
        assert torch.isfinite(starts[mask]).all()
        assert torch.isfinite(ends[mask]).all()


def test_activation_operations_are_distinct():
    values = torch.tensor([-2., -0.5, 0., 0.5, 2.])
    assert torch.equal(values, values)
    assert not torch.allclose(torch.tanh(values), F.gelu(values, approximate="none"))
    assert not torch.allclose(torch.tanh(values), values)
    assert not torch.allclose(F.gelu(values, approximate="none"), values)
