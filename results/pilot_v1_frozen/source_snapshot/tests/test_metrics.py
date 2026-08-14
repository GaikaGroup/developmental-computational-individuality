import pytest
torch = pytest.importorskip("torch")
from dci_pilot.metrics import causal_specialization, perm_matrix_distance, routing_specialization
def test_permutation_invariance():
    q1, q2 = torch.tensor([.8, .2]), torch.tensor([.2, .8])
    assert routing_specialization(q1, q2) == routing_specialization(q2, q1)
    c = torch.tensor([[.4, .1], [.1, .4]])
    assert causal_specialization(c) == causal_specialization(c.flip(0))
    assert perm_matrix_distance(c, c.flip(0)) == 0

