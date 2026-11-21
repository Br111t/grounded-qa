"""V5J interaction variants: shape, gradients and shared initialization."""
import torch

from experiments.question_aware_v5i.compare_interactions import InteractionSpanQA
from grounded_qa.models.span_model_v5 import prepare_span_input


def test_v5j_variants_forward_and_backward():
    ids, mask = prepare_span_input(
        "Fictional Workshop W0001. Leader: Fara. Location: Cedar 12. Topic: Python.",
        "What is the leader of Workshop W0001?",
    )
    for variant in ("multiplicative", "bilinear", "film"):
        torch.manual_seed(42)
        model = InteractionSpanQA(variant)
        starts, ends, answer = model(ids, mask)
        assert starts.shape == ids.shape
        assert ends.shape == ids.shape
        assert answer.shape == (1,)
        assert torch.isfinite(starts[mask]).all()
        assert torch.isfinite(ends[mask]).all()
        target = torch.nonzero(mask[0]).flatten()[0]
        loss = starts[0, target] + ends[0, target] + answer[0]
        loss.backward()
        if variant == "multiplicative":
            assert model.question_fusion.weight.grad is not None
        elif variant == "bilinear":
            assert model.start_bilinear.weight.grad is not None
            assert model.end_bilinear.weight.grad is not None
        else:
            assert model.film_projection.weight.grad is not None


def test_v5j_shared_encoder_initialization():
    models = []
    for variant in ("multiplicative", "bilinear", "film"):
        torch.manual_seed(42)
        models.append(InteractionSpanQA(variant))
    for key, tensor in models[0].state_dict().items():
        if key.startswith(("embedding.", "positions.", "encoder.",
                           "start_head.", "end_head.", "answerability_head.")):
            assert all(torch.equal(tensor, m.state_dict()[key]) for m in models[1:])


def test_v5j_film_initially_identity():
    torch.manual_seed(42)
    model = InteractionSpanQA("film").eval()
    assert torch.count_nonzero(model.film_projection.weight) == 0
    assert torch.count_nonzero(model.film_projection.bias) == 0
