from pathlib import Path

from dl_moe.experiments.dl_moe_01.protocol import experience_rows, seed_rows, validate_experience_rows
from dl_moe.experiments.dl_moe_01.statistics import paired_tost


def test_experience_manifest_has_identical_task_multiset_and_common_phase():
    seeds = {"development_data_seed": 10, "common_data_seed": 20}
    rows = experience_rows(1, seeds)
    validate_experience_rows(rows)
    assert len(rows) == 28000


def test_seed_manifest_has_30_primary_and_10_reserve_blocks():
    rows = seed_rows()
    assert len(rows) == 40
    assert sum(row["seed_role"] == "primary" for row in rows) == 30
    assert len({row["model_init_seed"] for row in rows}) == 40


def test_tost_equivalence_fixture():
    import numpy as np
    result = paired_tost(np.array([.001, -.002, .003, 0.0, .001]))
    assert result["equivalent"] is True


def test_corrupt_b_task_identity_is_rejected():
    import pytest
    rows = experience_rows(1, {"development_data_seed":10,"common_data_seed":20})
    row = next(r for r in rows if r['condition']=='BA' and r['task']=='B')
    row['batch_seed'] += 1
    with pytest.raises(ValueError,match='identities'):
        validate_experience_rows(rows)
