import pytest

torch = pytest.importorskip("torch")

from dci_pilot.compare_experiments import _paired_rows
from dci_pilot.metrics import perm_matrix_distance


def _row(seed, condition, value):
    return {
        "seed": seed,
        "condition": condition,
        "accuracy_A": .99,
        "accuracy_B": .99,
        "routing_A": [.8, .2],
        "routing_B": [.2, .8],
        "routing_specialization": .6,
        "router_entropy": .4,
        "causal_specialization": value,
        "ablation": {"expert_1": {"delta_A": value, "delta_B": 0}, "expert_2": {"delta_A": 0, "delta_B": value}},
        "router_collapse": {"collapsed": False},
    }


def test_paired_endpoint_aligns_triplets_without_pooling_models():
    rows = [_row(7, "INTERLEAVED", .1), _row(7, "BLOCKED_AB", .3), _row(7, "BLOCKED_BA", .5)]
    result = _paired_rows(rows)
    assert len(result) == 1
    assert result[0]["D_s"] == pytest.approx(.3)


def test_permutation_distance_is_expert_swap_invariant():
    a = torch.tensor([[.8, .2, .4, .1], [.2, .8, .1, .4]])
    swapped = a.flip(0)
    assert perm_matrix_distance(a, swapped) == pytest.approx(0)
