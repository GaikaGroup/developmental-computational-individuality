import csv
from pathlib import Path

import numpy as np
import pytest

from dl_moe.experiments.dl_moe_01.analyze import analyze
from dl_moe.experiments.dl_moe_01.statistics import paired_tost

PREREG = 'dl_moe/experiments/dl_moe_01/preregistration.yaml'


def fixture_tables(root, n=30, effect=0.02, accuracy=.99):
    root.mkdir(exist_ok=True)
    rows = [dict(seed_block_id=f'{i:03d}', primary_endpoint_available=True,
                 architecture_amplification_a=effect + (i % 3 - 1) * .003,
                 d_dl=.04 + (i % 4) * .002, d_pop_dl=.02 + (i % 5) * .001,
                 architecture_amplification_a_unrenorm=.01 + (i % 7) * .002)
            for i in range(1, n + 1)]
    behavior = [dict(seed_block_id=f'{i:03d}', architecture=a,
                     developmental_condition=c, task=t, checkpoint=20000,
                     balanced_accuracy=accuracy + (i % 3 - 1) * .001)
                for i in range(1, n + 1) for a in ('dl_moe', 'flat_moe')
                for c in ('INT', 'AB', 'BA') for t in ('A', 'B')]
    for name, table in [('seed_effects.csv', rows), ('behavior_metrics.csv', behavior)]:
        with (root / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=table[0]); w.writeheader(); w.writerows(table)
    return rows


def test_analysis_reads_schema_h5_and_applies_behavior_gate(tmp_path):
    fixture_tables(tmp_path)
    result = analyze(tmp_path, tmp_path / 'analysis', PREREG)
    assert result['strong_support'] is True
    with (tmp_path / 'analysis/statistical_summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {r['hypothesis'] for r in rows} == {'H1', 'H2', 'H4', 'H5'}
    assert all(float(r['p_value_adjusted']) >= float(r['p_value_raw']) for r in rows)


@pytest.mark.parametrize('n,effect,accuracy', [(27,.02,.99), (30,-.02,.99), (30,.02,.90)])
def test_incomplete_negative_and_low_performance_not_supported(tmp_path, n, effect, accuracy):
    fixture_tables(tmp_path, n, effect, accuracy)
    if n < 28:
        with pytest.raises(ValueError, match='incomplete'):
            analyze(tmp_path, tmp_path / 'analysis', PREREG)
    else:
        assert not analyze(tmp_path, tmp_path / 'analysis', PREREG)['strong_support']


def test_duplicate_block_rejected(tmp_path):
    fixture_tables(tmp_path)
    p = tmp_path / 'seed_effects.csv'
    with p.open('a') as f:
        f.write(p.read_text().splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        analyze(tmp_path, tmp_path / 'analysis', PREREG)


def test_tost_zero_variance_and_insufficient_sample():
    assert paired_tost(np.zeros(30))['equivalent']
    assert not paired_tost(np.array([0.]))['equivalent']
    assert not paired_tost(np.full(30, .02))['equivalent']


def test_holm_known_values():
    from dl_moe.experiments.dl_moe_01.statistics import holm
    assert holm([.03,.01,.04]) == pytest.approx([.06,.03,.06])


def test_missing_behavior_cannot_pass_gate(tmp_path):
    fixture_tables(tmp_path)
    p=tmp_path/'behavior_metrics.csv'
    lines=p.read_text().splitlines(); p.write_text('\n'.join(lines[:-1])+'\n')
    result=analyze(tmp_path,tmp_path/'analysis',PREREG)
    assert not result['behavioral_gate'] and not result['strong_support']


def test_nonfinite_endpoint_is_rejected(tmp_path):
    fixture_tables(tmp_path)
    p=tmp_path/'seed_effects.csv'
    p.write_text(p.read_text().replace('0.02,','nan,',1))
    with pytest.raises(ValueError,match='finite'):
        analyze(tmp_path,tmp_path/'analysis',PREREG)
