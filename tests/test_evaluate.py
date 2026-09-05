import pytest
torch = pytest.importorskip("torch")
from dci_pilot.evaluate import evaluate


class ConfidentAlternatingRouter(torch.nn.Module):
    def forward(self, x, knockout=None, return_router=False, knockout_mode="renormalized"):
        first = torch.where(x[:, :1] > 0, .99, .01)
        p = torch.cat((first, 1 - first), dim=1)
        logits = torch.zeros(len(x), device=x.device)
        return (logits, p) if return_router else logits


def test_evaluate_records_mean_per_example_entropy():
    result = evaluate(ConfidentAlternatingRouter(), "cpu", 10000, 71)
    assert result["router_entropy"] < .1
    assert result["router_entropy_of_task_means"] > .68
