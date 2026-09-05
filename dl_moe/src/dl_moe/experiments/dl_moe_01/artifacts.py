"""Freeze raw evidence and derive long-format tables without training."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .analyze import write_csv
from .execution import atomic_json, exclusive, now
from .protocol import json_hash, sha256_file
from .schemas import SCHEMAS


def raw_paths(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.name not in ('.execution.lock','raw_manifest.json') and not p.name.endswith('.tmp'))


def verify_raw(root):
    root=Path(root)
    manifest=json.loads((root/'raw_manifest.json').read_text())
    actual={str(p.relative_to(root)):sha256_file(p) for p in raw_paths(root)}
    if actual!=manifest['files']:
        raise ValueError('raw artifact integrity mismatch')
    return manifest


def freeze_raw(root):
    root=Path(root)
    with exclusive(root):
        if (root/'raw_manifest.json').exists():
            return verify_raw(root)
        if not (root/'completion.json').exists():
            raise ValueError('series is not terminal; cannot freeze raw artifacts')
        execution=json.loads((root/'execution.json').read_text())['identity']
        records=[json.loads(p.read_text()) for p in (root/'raw').glob('*/*/run.json')]
        expected={(b,a,c) for b in execution['blocks'] for a in ('dl_moe','flat_moe') for c in ('INT','AB','BA')}
        actual={(r['manifest']['seed_block_id'],r['manifest']['architecture'],r['manifest']['developmental_condition']) for r in records}
        if actual!=expected or len(records)!=len(expected) or any(r['manifest']['run_status'] not in ('completed','numerical_failure') for r in records):
            raise ValueError('incomplete run inventory')
        for p in (root/'raw').glob('*/*/run.json'):
            record=json.loads(p.read_text())
            if record['manifest']['run_status']=='completed':
                for step in execution['config']['evaluation']['checkpoints']:
                    for name in (f'step_{step}.pt',f'evaluation_{step}.json'):
                        if not (p.parent/name).is_file():
                            raise ValueError('completed run missing checkpoint or evaluation')
        manifest=dict(frozen_at=now(),technical=execution['technical'],files={str(p.relative_to(root)):sha256_file(p) for p in raw_paths(root)})
        atomic_json(root/'raw_manifest.json',manifest)
        return manifest


def aligned_distance(a,b,k=None):
    def squared(left,right):
        costs=((left[:,None,:]-right[None,:,:])**2).sum(-1)
        i,j=linear_sum_assignment(costs)
        return float(costs[i,j].sum())
    if k is None:
        distance=squared(a,b)
    else:
        distance=min(sum(squared(a[p*k:(p+1)*k],b[q*k:(q+1)*k]) for p,q in enumerate(order)) for order in ((0,1),(1,0)))
    return float(np.sqrt(distance/a.size))


def extract(root, output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output==root or root in output.parents:
        raise ValueError('derived tables must be outside the frozen raw directory')
    frozen=verify_raw(root)
    execution=json.loads((root/'execution.json').read_text())['identity']
    final=execution['config']['training']['total_steps']
    tables={name:[] for name in SCHEMAS if name not in ('behavioral_equivalence.csv','statistical_summary.csv')}
    profiles={}; summaries={}

    def add(name,base,**values):
        row={k:'' for k in SCHEMAS[name].split()}
        row.update({k:v for k,v in base.items() if k in row})
        if not set(values)<=set(row):
            raise ValueError(f'unknown schema fields: {set(values)-set(row)}')
        row.update(values); tables[name].append(row)

    for path in sorted((root/'raw').glob('*/*/run.json')):
        record=json.loads(path.read_text()); manifest=record['manifest']
        add('run_manifest.csv',manifest)
        base={k:manifest[k] for k in ('experiment_id','seed_block_id','architecture','developmental_condition','run_id')}
        for evaluation_path in sorted(path.parent.glob('evaluation_*.json')):
            evaluation=json.loads(evaluation_path.read_text()); step=evaluation['step']; metrics=evaluation['metrics']
            if evaluation['run_id']!=base['run_id'] or step not in execution['config']['evaluation']['checkpoints']:
                raise ValueError('evaluation identity mismatch')
            current=dict(base,checkpoint=step)
            for task,behavior in metrics['behavior'].items():
                add('behavior_metrics.csv',current,task=task,**behavior,eval_seed=evaluation['eval_seed'])
            for mode,effects in metrics['expert'].items():
                for e,effect in enumerate(effects):
                    for task,delta in effect.items():
                        normal=metrics['behavior'][task]['balanced_accuracy']
                        add('expert_knockout.csv',current,knockout_mode=mode,expert_id=e,
                            population_id=e//execution['config']['model']['experts_per_population'] if base['architecture']=='dl_moe' else '',
                            task=task,performance_normal=normal,performance_knockout=normal-delta,
                            delta_balanced_accuracy=delta,renormalized=mode=='renormalized')
            for mode,effects in metrics['population'].items():
                for p,effect in enumerate(effects):
                    for task,delta in effect.items():
                        normal=metrics['behavior'][task]['balanced_accuracy']
                        add('population_knockout.csv',current,population_id=p,task=task,knockout_mode=mode,
                            performance_normal=normal,performance_knockout=normal-delta,delta_balanced_accuracy=delta)
            for intervention,effect in metrics['communication_knockout'].items():
                for task,delta in effect.items():
                    normal=metrics['behavior'][task]['balanced_accuracy']
                    add('communication_knockout.csv',current,intervention=intervention,task=task,
                        performance_normal=normal,performance_intervention=normal-delta,delta_balanced_accuracy=delta)
            for row in metrics['routing']:
                add('routing_diagnostics.csv',current,**row)
            for row in metrics['communication']:
                add('communication_diagnostics.csv',current,**row)
            def specialization(effects):
                return float(np.mean([abs(e['A']-e['B']) for e in effects]))
            profile=np.array([[e['A'],e['B']] for e in metrics['expert']['renormalized']])
            summary=dict(balanced_accuracy_A=metrics['behavior']['A']['balanced_accuracy'],
                         balanced_accuracy_B=metrics['behavior']['B']['balanced_accuracy'],
                         mean_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in metrics['behavior'].values()])),
                         s_expert_causal_renorm=specialization(metrics['expert']['renormalized']),
                         s_expert_causal_unrenorm=specialization(metrics['expert']['unrenormalized']),
                         s_population_causal=specialization(metrics['population']['renormalized']) if metrics['population'] else '',
                         organizational_profile_hash=json_hash(profile.tolist()))
            add('run_summary.csv',current,**summary)
            key=(base['seed_block_id'],base['architecture'],base['developmental_condition'],step)
            profiles[key]=(profile,base['run_id'])
            if step==final and manifest['run_status']=='completed':
                summaries[key[:3]]=summary
    for block in execution['blocks']:
        complete=all((block,a,c) in summaries for a in ('dl_moe','flat_moe') for c in ('INT','AB','BA'))
        effect=dict(primary_endpoint_available=complete,behavior_equivalence_complete=complete)
        if complete:
            for suffix,metric in [('', 's_expert_causal_renorm'),('_unrenorm','s_expert_causal_unrenorm')]:
                for label,arch in [('dl','dl_moe'),('flat','flat_moe')]:
                    vals={c:summaries[(block,arch,c)][metric] for c in ('INT','AB','BA')}
                    effect.update({f's_{label}_{c.lower()}{suffix}':v for c,v in vals.items()})
                    effect[f'd_{label}{suffix}']=(vals['AB']+vals['BA'])/2-vals['INT']
                effect[f'architecture_amplification_a{suffix}']=effect[f'd_dl{suffix}']-effect[f'd_flat{suffix}']
            vals={c:summaries[(block,'dl_moe',c)]['s_population_causal'] for c in ('INT','AB','BA')}
            effect.update({f's_pop_dl_{c.lower()}':v for c,v in vals.items()})
            effect['d_pop_dl']=(vals['AB']+vals['BA'])/2-vals['INT']
        add('seed_effects.csv',dict(experiment_id='DL-MoE-01',seed_block_id=block),**effect)
        for arch,step,(ca,cb) in itertools.product(('dl_moe','flat_moe'),execution['config']['evaluation']['checkpoints'],itertools.combinations(('INT','AB','BA'),2)):
            if (block,arch,ca,step) not in profiles or (block,arch,cb,step) not in profiles:
                continue
            a,run_a=profiles[(block,arch,ca,step)]; b,run_b=profiles[(block,arch,cb,step)]
            modes=['global','within_population'] if arch=='dl_moe' else ['global']
            for mode in modes:
                add('organizational_distance.csv',dict(experiment_id='DL-MoE-01',seed_block_id=block,architecture=arch,checkpoint=step),
                    comparison=f'{ca}_vs_{cb}',raw_distance=float(np.linalg.norm(a-b)/np.sqrt(a.size)),
                    aligned_distance=aligned_distance(a,b,execution['config']['model']['experts_per_population'] if mode=='within_population' else None),
                    alignment_mode=mode,population_swap_allowed=arch=='dl_moe',model_a_run_id=run_a,model_b_run_id=run_b)
    import csv
    for name in ('protocol_hashes.csv','exclusions_and_incidents.csv'):
        if (root/name).exists():
            with (root/name).open() as f:
                tables[name]=list(csv.DictReader(f))
    # Validate scalar finiteness before writing any derived table.
    for table in tables.values():
        for row in table:
            if any(isinstance(v,float) and not np.isfinite(v) for v in row.values()):
                raise ValueError('nonfinite derived metric')
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        write_csv(output/name,rows,SCHEMAS[name].split())
    import shutil
    shutil.copyfile(root/'seed_manifest.csv',output/'seed_manifest.csv')
    atomic_json(output/'extraction.json',dict(technical=frozen['technical'],raw_manifest_sha256=sha256_file(root/'raw_manifest.json'),
        source_raw=str(root),primary_checkpoint=final,table_hashes={name:sha256_file(output/name) for name in (*tables,'seed_manifest.csv')}))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True); p.add_argument('--output')
    p.add_argument('--freeze',action='store_true'); a=p.parse_args(argv)
    if a.freeze:
        freeze_raw(a.raw)
    if a.output:
        extract(a.raw,a.output)
    elif not a.freeze:
        verify_raw(a.raw)


if __name__=='__main__':
    main()
