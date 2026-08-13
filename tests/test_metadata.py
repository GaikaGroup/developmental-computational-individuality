import pytest
torch = pytest.importorskip("torch")
from dci_pilot.model import TinySoftMoE
from dci_pilot.utils import metadata


def test_resolved_device_is_not_overwritten_by_auto():
    result = metadata(1, 2, "INTERLEAVED", "cpu", TinySoftMoE(), {"device": "auto"})
    assert result["configured_device"] == "auto"
    assert result["device"] == "cpu"
