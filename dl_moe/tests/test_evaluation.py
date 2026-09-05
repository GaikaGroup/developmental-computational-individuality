import pytest

torch = pytest.importorskip("torch")

from dl_moe.evaluation import evaluate_causal
from dl_moe.model import DLMoE, DLMoEConfig


def test_causal_evaluation_returns_all_intervention_profiles():
    model = DLMoE(DLMoEConfig(model_dim=16, expert_hidden_dim=24, experts_per_population=2, comm_dim=4))
    result = evaluate_causal(model, "cpu", n=32, seed=17)
    assert result["expert_causal_effects"].shape == (4, 2)
    assert result["population_causal_effects"].shape == (2, 2)
    assert set(result["communication_effects"]) == {"0_to_1", "1_to_0", "both"}
