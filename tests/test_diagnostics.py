from dci_pilot.metrics import prolonged_router_collapse


def test_prolonged_collapse_requires_same_expert_and_duration():
    history = [
        {"step": step, "routing_A": route, "routing_B": route}
        for step, route in ((0, [.96, .04]), (1000, [.97, .03]), (2000, [.98, .02]))
    ]
    result = prolonged_router_collapse(history)
    assert result["collapsed"] and result["expert"] == 1


def test_short_router_excursion_is_not_prolonged_collapse():
    history = [
        {"step": step, "routing_A": [.99, .01], "routing_B": [.99, .01]}
        for step in (0, 500, 1000)
    ]
    assert not prolonged_router_collapse(history)["collapsed"]
