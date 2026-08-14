import pytest
torch = pytest.importorskip("torch")
from dci_pilot.analyze import _organization_matrix
from dci_pilot.metrics import perm_matrix_distance

def test_organization_distance_ignores_expert_labels():
    row = {"routing_A": [.8, .2], "routing_B": [.2, .8], "ablation": {"expert_1": {"delta_A": .4, "delta_B": .1}, "expert_2": {"delta_A": .1, "delta_B": .4}}}
    swapped = {"routing_A": [.2, .8], "routing_B": [.8, .2], "ablation": {"expert_1": {"delta_A": .1, "delta_B": .4}, "expert_2": {"delta_A": .4, "delta_B": .1}}}
    assert perm_matrix_distance(_organization_matrix(row), _organization_matrix(swapped)) == 0
