import pytest

torch = pytest.importorskip("torch")

from dl_moe.compatibility import s_causal_original
from dl_moe.metrics import causal_specialization_general, organizational_distance, s_expert_causal


def test_legacy_metric_is_exact_and_not_overloaded():
    matrix = torch.tensor([[0.4, 0.1], [0.1, 0.4]])
    assert s_causal_original(matrix) == pytest.approx(0.3)
    assert causal_specialization_general(matrix) == pytest.approx(0.15)
    assert s_expert_causal(matrix) == pytest.approx(s_causal_original(matrix))
    with pytest.raises(ValueError):
        s_causal_original(torch.zeros(4, 2))


def test_s_expert_causal_uses_preregistered_eight_expert_formula():
    matrix = torch.tensor([[.4, .1], [.1, .4], [.0, .2], [.2, .0]])
    assert s_expert_causal(matrix) == pytest.approx(.25)


def test_organizational_distance_respects_permutation():
    matrix = torch.tensor([[0.4, 0.1], [0.1, 0.4], [0.2, 0.3], [0.3, 0.2]])
    assert organizational_distance(matrix, matrix.flip(0), experts_per_population=2, mode="within_population") == pytest.approx(0.0)
    changed = matrix.clone(); changed[0, 0] += 0.5
    assert organizational_distance(matrix, changed) > 0
