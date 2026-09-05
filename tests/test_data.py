import pytest
torch = pytest.importorskip("torch")
from dci_pilot.data import make_task
def test_tasks_are_balanced_and_deterministic():
    for task, seed in (("A", 1), ("B", 2)):
        x1, y1 = make_task(task, 20000, seed); x2, y2 = make_task(task, 20000, seed)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert abs(float(y1.mean()) - .5) < .02

