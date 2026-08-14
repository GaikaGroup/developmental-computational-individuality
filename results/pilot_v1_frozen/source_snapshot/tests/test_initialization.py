import pytest
torch = pytest.importorskip("torch")
from dci_pilot.model import TinySoftMoE
from dci_pilot.utils import seed_everything, state_checksum

def test_paired_initialization_is_exactly_equal():
    states = []
    for _ in range(3):
        seed_everything(17); model = TinySoftMoE(); states.append(model.state_dict())
    checksums = [state_checksum(s) for s in states]
    assert len(set(checksums)) == 1
    assert all(torch.equal(states[0][k], states[i][k]) for i in (1, 2) for k in states[0])

