from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ...training import load_config
from .statistics import holm, one_sided_t, paired_tost, sign_flip
from .protocol import seed_rows, sha256_file


def _stats(values, bootstrap_seed=0, samples=10000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('statistics require at least two finite seed-level values')
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    sd = float(values.std(ddof=1))
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                sd=sd, se=sd/np.sqrt(len(values)), ci95_lower=float(np.quantile(bootstrap,.025)),
                ci95_upper=float(np.quantile(bootstrap,.975)),
                cohens_dz=float(values.mean()/sd) if sd else None,
                positive=int((values>0).sum()), zero=int((values==0).sum()), negative=int((values<0).sum()))


def read_csv(path):
    with Path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze(results, output, preregistration, *, technical=False):
    results, output = Path(results), Path(output)
    if results.resolve() == output.resolve():
        raise ValueError('analysis output must be separate from source tables')
    from .validate_preregistration import validate
    if errors := validate(prereg_path=preregistration):
        raise ValueError('; '.join(errors))
    prereg = load_config(preregistration)
    if prereg['experiment_id'] != 'DL-MoE-01':
        raise ValueError('wrong experiment preregistration')
    provenance = results / 'extraction.json'
    extraction = json.loads(provenance.read_text()) if provenance.exists() else {}
    if extraction.get('technical') and not technical:
        raise ValueError('technical artifacts cannot enter confirmatory analysis')
    for name, digest in extraction.get('table_hashes', {}).items():
        if sha256_file(results / name) != digest:
            raise ValueError('derived table integrity mismatch')
    rows = read_csv(results / 'seed_effects.csv')
    ids = [r['seed_block_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate seed block')
    expected = {r['seed_block_id'] for r in seed_rows() if r['seed_role'] == 'primary'}
    if not technical and not set(ids) <= expected:
        raise ValueError('unexpected or reserve seed block')
    primary = sorted([r for r in rows if r['primary_endpoint_available'].lower() == 'true'], key=lambda r:r['seed_block_id'])
    minimum = 2 if technical else prereg['exclusion_rules']['minimum_complete_primary_blocks']
    if len(primary) < minimum:
        raise ValueError(f'confirmatory analysis incomplete: {len(primary)} complete blocks, minimum is {minimum}')
    alpha = prereg['statistical_tests']['alpha']
    if prereg['multiple_testing']['secondary_correction'] != 'holm':
        raise ValueError('unsupported secondary correction')
    statistical_rows = []
    for hypothesis, column in [('H2','architecture_amplification_a'), ('H1','d_dl'),
                               ('H4','d_pop_dl'), ('H5','architecture_amplification_a_unrenorm')]:
        vector = np.asarray([float(r[column]) for r in primary])
        stats = _stats(vector, samples=prereg['statistical_tests']['bootstrap_samples'])
        test = one_sided_t(vector)
        statistical_rows.append(dict(hypothesis=hypothesis, endpoint=column, analysis_set_n=len(vector),
            mean=stats['mean'], median=stats['median'], sd=stats['sd'], standard_error=stats['se'],
            ci95_lower=stats['ci95_lower'], ci95_upper=stats['ci95_upper'], effect_size_name='cohens_dz',
            effect_size=stats['cohens_dz'], test_name='one_sample_t', alternative='greater',
            test_statistic=test['statistic'], p_value_raw=test['p_value'], p_value_adjusted=test['p_value'],
            positive_count=stats['positive'], zero_count=stats['zero'], negative_count=stats['negative'],
            supported=False, notes='All complete primary blocks; no outcome exclusions.'))
    for row, adjusted in zip(statistical_rows[1:], holm([r['p_value_raw'] for r in statistical_rows[1:]])):
        row['p_value_adjusted'] = adjusted
    for row in statistical_rows:
        row['supported'] = bool(row['mean'] > 0 and row['p_value_adjusted'] < alpha)
    behavior = read_csv(results / 'behavior_metrics.csv')
    checkpoint = str(extraction.get('primary_checkpoint', prereg['primary_endpoint']['checkpoint']) if technical else prereg['primary_endpoint']['checkpoint'])
    by_key = {}
    for row in behavior:
        if str(row['checkpoint']) != checkpoint:
            continue
        key = (row['architecture'],row['seed_block_id'],row['developmental_condition'],row['task'])
        if key in by_key:
            raise ValueError('duplicate behavior row')
        value = float(row['balanced_accuracy'])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid balanced accuracy')
        if key[0] not in ('dl_moe','flat_moe') or key[2] not in ('INT','AB','BA') or key[3] not in ('A','B') or key[1] not in ids:
            raise ValueError('unexpected behavior identity')
        by_key[key] = value
    block_ids = [r['seed_block_id'] for r in primary]
    equivalence_rows, floors = [], []
    behavior_complete = True
    margin = prereg['equivalence_margin']
    for architecture in ('dl_moe','flat_moe'):
        for task in ('A','B'):
            for condition in ('INT','AB','BA'):
                # Retain all available primary outcomes, including incomplete blocks.
                observations = [v for (a,b,c,t),v in by_key.items() if a==architecture and c==condition and t==task]
                mean = float(np.mean(observations)) if observations else None
                floors.append(dict(architecture=architecture,condition=condition,task=task,n=len(observations),
                    mean=mean,passes=bool(mean is not None and mean >= prereg['performance_floor'])))
            for blocked in ('AB','BA'):
                complete = all((architecture,b,c,task) in by_key for b in block_ids for c in (blocked,'INT'))
                behavior_complete &= complete
                if not complete:
                    continue
                differences = np.array([by_key[(architecture,b,blocked,task)]-by_key[(architecture,b,'INT',task)] for b in block_ids])
                equivalence_rows.append(dict(architecture=architecture,task=task,comparison=f'{blocked}_minus_INT',
                    **paired_tost(differences,margin),sd_difference=float(differences.std(ddof=1)),
                    equivalence_margin_lower=-margin,equivalence_margin_upper=margin))
    behavioral_gate = bool(behavior_complete and len(equivalence_rows)==8 and
                          all(r['equivalent'] for r in equivalence_rows) and all(r['passes'] for r in floors))
    primary_stats = _stats(np.array([float(r['architecture_amplification_a']) for r in primary]),
                           samples=prereg['statistical_tests']['bootstrap_samples'])
    primary_stats.update(t_test=one_sided_t(np.array([float(r['architecture_amplification_a']) for r in primary])),
                         sign_flip_p=sign_flip(np.array([float(r['architecture_amplification_a']) for r in primary]),0),
                         sign_flip_alternative='two-sided', analysis_random_seed=0)
    support = bool(statistical_rows[0]['supported'] and primary_stats['ci95_lower'] > 0 and behavioral_gate)
    summary = dict(experiment_id='DL-MoE-01',preregistration=str(preregistration),
        preregistration_sha256=sha256_file(preregistration),primary=primary_stats,analysis_set_n=len(primary),
        incomplete_or_missing_blocks=sorted(expected-set(block_ids)) if not technical else [],
        behavioral_gate=behavioral_gate,performance_floor=floors,strong_support=support and not technical,
        technical=technical,trained=False,
        input_hashes={name:sha256_file(results/name) for name in ('seed_effects.csv','behavior_metrics.csv')})
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'statistical_summary.csv',statistical_rows)
    from .schemas import SCHEMAS
    write_csv(output/'behavioral_equivalence.csv',equivalence_rows,SCHEMAS['behavioral_equivalence.csv'].split())
    (output/'analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'DL_MOE_01_INTERPRETATION.md').write_text(
        f"# DL-MoE-01 interpretation\n\nTechnical validation: {technical}. Complete blocks: {len(primary)}.\n\n"
        f"H2 statistical support: {statistical_rows[0]['supported']}. Behavioral gate: {behavioral_gate}. "
        f"Strong confirmatory support: {summary['strong_support']}.\n\n"
        'Incomplete blocks remain recorded. Numerical failures are not replaced. '
        'No comparable-behavior claim is allowed when the behavioral gate fails.\n')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True); p.add_argument('--preregistration',required=True)
    p.add_argument('--output',required=True); p.add_argument('--technical',action='store_true')
    a=p.parse_args(argv)
    print(json.dumps(analyze(a.results,a.output,a.preregistration,technical=a.technical),indent=2))


if __name__ == '__main__':
    main()
