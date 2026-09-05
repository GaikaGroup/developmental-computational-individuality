import pytest

torch = pytest.importorskip("torch")

from dl_moe.model import DLMoE, DLMoEConfig, FlatMoE, Intervention


def test_forward_shapes_and_gradients():
    model = DLMoE(DLMoEConfig(model_dim=16, expert_hidden_dim=24, experts_per_population=2, comm_dim=4))
    x = torch.randn(5, 18)
    output = model(x, return_diagnostics=True)
    assert output.logits.shape == (5,)
    assert output.population_probs.shape == (5, 2)
    assert output.expert_probs.shape == (5, 2, 2)
    output.logits.mean().backward()
    assert model.population_router.weight.grad is not None
    assert model.expert_routers[1].weight.grad is not None


@pytest.mark.parametrize("mode", ["soft", "top1", "top2"])
def test_routing_modes_and_communication_knockout(mode):
    cfg = DLMoEConfig(model_dim=16, expert_hidden_dim=24, experts_per_population=3, comm_dim=4, routing_mode=mode, top_k=1 if mode != "top2" else 2)
    model = DLMoE(cfg)
    output = model(torch.randn(4, 18), Intervention(disable_comm_0_to_1=True), return_diagnostics=True)
    assert output.logits.shape == (4,)
    assert torch.allclose(output.expert_probs.sum(-1), torch.ones(4, 2))


def test_knockouts_do_not_mutate_parameters_and_comm_zero_works():
    model = DLMoE(DLMoEConfig(model_dim=16, expert_hidden_dim=24, comm_dim=0))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model(torch.randn(3, 18), Intervention(disabled_experts=((0, 0),), renormalize=True))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert model(torch.randn(3, 18)).shape == (3,)


def test_flat_control_has_eight_experts_by_default():
    model = FlatMoE(DLMoEConfig(model_dim=16, expert_hidden_dim=24))
    assert model.total_experts == 8
