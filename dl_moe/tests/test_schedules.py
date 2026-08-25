from dl_moe.schedules import make_schedule


def test_schedules_have_equal_exposure_and_common_mature_phase():
    schedules = {name: make_schedule(name, early_steps=10, total_steps=20, seed=3) for name in ("interleaved", "blocked_ab", "blocked_ba")}
    assert all(schedule.tasks.count("A") == schedule.tasks.count("B") == 10 for schedule in schedules.values())
    assert schedules["blocked_ab"].early == ("A",) * 5 + ("B",) * 5
    assert schedules["blocked_ba"].early == ("B",) * 5 + ("A",) * 5
    assert schedules["interleaved"].mature == schedules["blocked_ab"].mature == schedules["blocked_ba"].mature
