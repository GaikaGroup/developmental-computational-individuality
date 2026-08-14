import pytest
torch = pytest.importorskip("torch")
from dci_pilot.model import TinySoftMoE
def test_router_and_knockout():
    m = TinySoftMoE(); x = torch.randn(4, 18); logits, p = m(x, return_router=True)
    assert logits.shape == (4,) and torch.allclose(p.sum(1), torch.ones(4))
    assert m(x, knockout=0).shape == (4,)

