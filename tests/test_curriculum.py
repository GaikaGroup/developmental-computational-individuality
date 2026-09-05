import pytest
pytest.importorskip("torch")
from dci_pilot.curriculum import build_curriculum
def test_curricula_match_exposure_and_order():
    cs = {c: build_curriculum(c, 10, 20, 3) for c in ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA")}
    assert all(c.tasks.count("A") == c.tasks.count("B") == 10 for c in cs.values())
    assert cs["BLOCKED_AB"].early == ("A",) * 5 + ("B",) * 5
    assert cs["BLOCKED_BA"].early == ("B",) * 5 + ("A",) * 5
    assert cs["INTERLEAVED"].mature == cs["BLOCKED_AB"].mature == cs["BLOCKED_BA"].mature

