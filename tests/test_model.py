import pytest
torch = pytest.importorskip("torch")
from dci_pilot.model import TinySoftMoE
def test_router_and_knockout():
    m = TinySoftMoE(); x = torch.randn(4, 18); logits, p = m(x, return_router=True)
    assert logits.shape == (4,) and torch.allclose(p.sum(1), torch.ones(4))
    assert m(x, knockout=0).shape == (4,)

def test_both_knockout_definitions_match_their_formulas():
    m = TinySoftMoE(); x = torch.randn(4, 18)
    with torch.no_grad():
        h = m.encoder(x); p = m.router(h).softmax(-1); survivor = m.experts[1](h)
        expected_renormalized = m.head(m.norm(h + survivor)).squeeze(-1)
        expected_non_renormalized = m.head(m.norm(h + p[:, 1:2] * survivor)).squeeze(-1)
    assert torch.allclose(m(x, knockout=0), expected_renormalized)
    assert torch.allclose(m(x, knockout=0, knockout_mode="non_renormalized"), expected_non_renormalized)
