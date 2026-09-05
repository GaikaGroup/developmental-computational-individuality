"""Deterministic execution and durable raw artifacts for DL-MoE-01."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from ...compatibility import make_dci_task
from ...model import DLMoE, DLMoEConfig, FlatMoE
from .analyze import write_csv
from .evaluate import evaluate_checkpoint
from .protocol import experience_rows, json_hash, seed_rows, sha256_file
from .schemas import SCHEMAS
from .validate_preregistration import REPO

ARCHITECTURES=('dl_moe','flat_moe')
CONDITIONS=('INT','AB','BA')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def exclusive(root):
    # OS lock is released on process exit; a killed process cannot leave a stale lock.
    import fcntl
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/'.execution.lock').open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process owns this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def source_hash():
    paths=sorted((REPO/'dl_moe/src').rglob('*.py'))
    return json_hash({str(p.relative_to(REPO)):sha256_file(p) for p in paths})


def runtime():
    return dict(python=platform.python_version(),platform=platform.platform(),torch=str(torch.__version__),
                cuda=torch.version.cuda,packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),
                torch_threads=torch.get_num_threads())


def git_state(require_clean=False):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=REPO,text=True)
    if require_clean and dirty:
        raise ValueError('confirmatory execution requires a clean committed checkout (including untracked files)')
    tags=subprocess.check_output(['git','tag','--points-at','HEAD'],cwd=REPO,text=True).splitlines()
    if require_clean and not tags:
        raise ValueError('confirmatory execution requires a tag on the reviewed commit')
    return dict(commit=commit,dirty=bool(dirty),tags=tags)


def plan_rows(config, block, technical=False):
    rows=experience_rows(int(block['seed_block_id']),block)
    if technical:
        early=config['training']['early_steps']; common=config['training']['total_steps']-early
        selected=[]
        for condition in CONDITIONS:
            subset=[r for r in rows if r['phase']=='developmental' and r['condition']==condition and int(r['batch_id'][1:])<=early//2]
            for i,row in enumerate(subset,1):
                selected.append(dict(row,step=i))
        selected += [dict(r) for r in rows if r['phase']=='common' and r['step']<=common]
        rows=selected
    return rows


def parameter_groups(model):
    counts=dict(encoder=0,experts=0,routers=0,communication=0,head=0)
    for name,p in model.named_parameters():
        group=('encoder' if name.startswith('encoder') else 'experts' if name.startswith('experts')
               else 'routers' if 'router' in name else 'communication' if name.startswith(('comm_','communication_')) else 'head')
        counts[group]+=p.numel()
    # head includes output/population normalizations; accounting covers every parameter.
    return {'parameter_count_'+k:v for k,v in counts.items()} | {'parameter_count_total':sum(counts.values())}


def refresh_manifest(root):
    rows=[]
    for p in sorted((root/'raw').glob('*/*/run.json')):
        record=json.loads(p.read_text())
        rows.append(record['manifest'])
    write_csv(root/'run_manifest.csv',rows,SCHEMAS['run_manifest.csv'].split())


def incident(root, manifest, exc, numerical):
    path=root/'exclusions_and_incidents.csv'
    import csv
    rows=list(csv.DictReader(path.open())) if path.exists() else []
    rows.append(dict(seed_block_id=manifest['seed_block_id'],run_id=manifest['run_id'],architecture=manifest['architecture'],
        developmental_condition=manifest['developmental_condition'],incident_type=type(exc).__name__,incident_time=now(),
        scientific_or_infrastructure='scientific' if numerical else 'infrastructure',excluded_from_primary=False,
        reason=str(exc),action_taken='retain numerical failure' if numerical else 'resume same seed/checkpoint',
        replacement_seed_used=False,deviation_id=''))
    write_csv(path,rows,SCHEMAS['exclusions_and_incidents.csv'].split())


def run_model(root, config, block, architecture, condition, rows, identity, checkpoint_interval=500, stop_after=None):
    run_id=f"{block['seed_block_id']}_{architecture}_{condition}"
    directory=root/'raw'/block['seed_block_id']/f'{architecture}_{condition}'
    directory.mkdir(parents=True,exist_ok=True)
    record_path=directory/'run.json'
    if record_path.exists():
        record=json.loads(record_path.read_text())
        if record['identity']!=identity:
            raise ValueError('run identity mismatch')
        if record['manifest']['run_status'] in ('completed','numerical_failure'):
            return
    device=config['device']
    torch.manual_seed(block['model_init_seed'])
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(block['model_init_seed'])
    model=(DLMoE if architecture=='dl_moe' else FlatMoE)(DLMoEConfig(**config['model'])).to(device=device,dtype=torch.float32)
    initial=state_hash(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    manifest={k:'' for k in SCHEMAS['run_manifest.csv'].split()}
    manifest.update(experiment_id='DL-MoE-01',preregistration_version='1.0',seed_block_id=block['seed_block_id'],
        seed_role=block['seed_role'],architecture=architecture,developmental_condition=condition,run_id=run_id,
        **{k:v for k,v in block.items() if k.endswith('_seed') and k in manifest},
        experience_manifest_sha256=sha256_file(root/'experience'/f"{block['seed_block_id']}.csv"),
        initial_state_sha256=initial,config_sha256=json_hash(config),code_commit=identity['git']['commit'],
        **parameter_groups(model),device=device,torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        python_version=platform.python_version(),run_status='running',start_time=now(),end_time='',wall_time_seconds=0,
        final_checkpoint_available=False,primary_endpoint_available=False,router_collapse_flag=False,
        numerical_failure_flag=False,infrastructure_incident_flag=False,
        notes='technical validation only' if identity['technical'] else '')
    record=dict(identity=identity,manifest=manifest)
    if record_path.exists():
        record=json.loads(record_path.read_text()); manifest=record['manifest']; manifest['run_status']='running'
    else:
        save_checkpoint(directory/'initial.pt',{'model':model.state_dict(),'initial_state_sha256':initial})
    atomic_json(record_path,record); refresh_manifest(root)
    sequence=[r for r in rows if r['condition']==condition and r['phase']=='developmental']+[r for r in rows if r['phase']=='common']
    last=0
    latest=directory/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=True)
        if saved['identity_hash']!=json_hash(identity) or saved['initial_state_sha256']!=initial:
            raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); last=saved['step']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([r.cpu() for r in saved['rng_cuda']])
    started=time.monotonic()

    def checkpoint(step):
        saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                   rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                   identity_hash=json_hash(identity),initial_state_sha256=initial)
        save_checkpoint(latest,saved)
        if step in config['evaluation']['checkpoints']:
            save_checkpoint(directory/f'step_{step}.pt',saved)

    def evaluate(step):
        path=directory/f'evaluation_{step}.json'
        if not path.exists():
            evaluation=evaluate_checkpoint(model,config['evaluation']['examples'],block['eval_seed'])
            atomic_json(path,dict(run_id=run_id,step=step,eval_seed=block['eval_seed'],metrics=evaluation))
        evaluation=json.loads(path.read_text())['metrics']
        manifest['router_collapse_flag'] |= any(r['router_collapse_flag'] for r in evaluation['routing'])

    try:
        if last in config['evaluation']['checkpoints']:
            archive = directory / f'step_{last}.pt'
            if not archive.exists():
                save_checkpoint(archive, saved)
            evaluate(last)
        for step,row in enumerate(sequence,1):
            if step<=last:
                continue
            model.train()
            x,y=make_dci_task(row['task'],config['training']['batch_size'],row['batch_seed'],device)
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite training loss at step {step}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'nonfinite gradient at step {step}')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError(f'nonfinite parameter at step {step}')
            if step%checkpoint_interval==0 or step in config['evaluation']['checkpoints'] or step==stop_after:
                checkpoint(step)
                atomic_json(root/'progress.json',dict(run_id=run_id,step=step,total_steps=len(sequence),timestamp=now()))
            if step==stop_after:
                raise InterruptedError('technical injected interruption')
            if step in config['evaluation']['checkpoints']:
                evaluate(step)
        manifest.update(run_status='completed',final_checkpoint_available=True,primary_endpoint_available=True)
    except FloatingPointError as exc:
        manifest.update(run_status='numerical_failure',numerical_failure_flag=True)
        incident(root,manifest,exc,True)
    except BaseException as exc:
        manifest.update(run_status='interrupted',infrastructure_incident_flag=True)
        incident(root,manifest,exc,False)
        raise
    finally:
        manifest['final_checkpoint_available']=(directory/f"step_{config['training']['total_steps']}.pt").exists()
        manifest['end_time']=now(); manifest['wall_time_seconds']+=time.monotonic()-started
        atomic_json(record_path,record); refresh_manifest(root)


def ensure_execution_artifacts(root, identity):
    """Idempotent initialization after a crash, without replacing mismatched evidence."""
    import csv
    from .validate_preregistration import DEFAULT_CONFIG, DEFAULT_MANIFEST, DEFAULT_PREREG

    def ensure_bytes(path, expected):
        if path.exists():
            if path.read_bytes() != expected:
                raise ValueError(f'execution artifact changed: {path.name}')
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    ensure_bytes(root / 'environment.lock.txt',
                 ('\n'.join(identity['runtime']['packages']) + '\n').encode())
    paths = [DEFAULT_CONFIG, DEFAULT_MANIFEST, DEFAULT_PREREG,
             REPO / 'docs/preregistration/DL_MOE_01_PREREGISTRATION.md']
    for source in paths:
        ensure_bytes(root / 'protocol' / source.name, source.read_bytes())
    ensure_bytes(root / 'seed_manifest.csv', DEFAULT_MANIFEST.read_bytes())
    expected = [dict(artifact=p.name, path=str(p.relative_to(REPO)), sha256=sha256_file(p))
                for p in paths]
    manifest = root / 'protocol_hashes.csv'
    if manifest.exists():
        with manifest.open() as handle:
            actual = [{k: row[k] for k in ('artifact', 'path', 'sha256')}
                      for row in csv.DictReader(handle)]
        if actual != expected:
            raise ValueError('execution protocol hashes changed')
    else:
        write_csv(manifest, [dict(row, created_at=now()) for row in expected],
                  SCHEMAS['protocol_hashes.csv'].split())


def execute(config, root, *, technical=False, blocks=None, resume=False, stop_after=None):
    import copy
    config=copy.deepcopy(config); root=Path(root)
    if checkpoint_steps := config['evaluation']['checkpoints']:
        if checkpoint_steps[-1] != config['training']['total_steps']:
            raise ValueError('final checkpoint required')
    else:
        raise ValueError('checkpoints required')
    if config['device']=='auto':
        config['device']='cuda' if torch.cuda.is_available() else 'cpu'
    if config['device'] not in ('cpu','cuda'):
        raise ValueError('validated execution devices are cpu and cuda')
    if not technical and blocks is not None:
        raise ValueError('confirmatory series always plans all 30 primary blocks')
    selected=[b for b in seed_rows() if b['seed_role']=='primary']
    if technical:
        if blocks is not None and not 1 <= blocks <= 30:
            raise ValueError('technical block count must be between 1 and 30')
        selected=selected[:blocks or 2]
    identity=dict(experiment_id='DL-MoE-01',technical=technical,config=config,source_hash=source_hash(),
                  git=git_state(require_clean=not technical),runtime=runtime(),blocks=[r['seed_block_id'] for r in selected])
    if not technical:
        from .validate_preregistration import validate
        if errors:=validate():
            raise ValueError('; '.join(errors))
        from ...training import load_config
        from .validate_preregistration import DEFAULT_CONFIG
        frozen=load_config(DEFAULT_CONFIG); frozen['device']=config['device']
        if config!=frozen:
            raise ValueError('execution config differs from frozen protocol')
    torch.use_deterministic_algorithms(True)
    if config['device']=='cuda':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
            raise ValueError('CUDA deterministic execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup')
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    with exclusive(root):
        if (root/'raw_manifest.json').exists():
            raise ValueError('raw results are frozen; execution is forbidden')
        protocol=root/'execution.json'
        if protocol.exists():
            previous=json.loads(protocol.read_text())
            if previous['identity']!=identity:
                raise ValueError('execution identity changed; cannot resume')
            if not resume:
                raise ValueError('output already exists; use explicit resume')
        else:
            if resume:
                raise ValueError('no execution to resume')
            if any(root.iterdir()) and set(p.name for p in root.iterdir()) != {'.execution.lock'}:
                raise ValueError('output must be empty for a new execution')
            atomic_json(protocol,dict(identity=identity,started_at=now()))
        ensure_execution_artifacts(root, identity)
        for block in selected:
            rows=plan_rows(config,block,technical)
            path=root/'experience'/f"{block['seed_block_id']}.csv"
            if not path.exists():
                write_csv(path,rows)
            else:
                import csv
                with path.open() as f:
                    actual=list(csv.DictReader(f))
                if actual != [{k:str(v) for k,v in r.items()} for r in rows]:
                    raise ValueError('experience manifest changed')
            for architecture in ARCHITECTURES:
                for condition in CONDITIONS:
                    run_model(root,config,block,architecture,condition,rows,identity,stop_after=stop_after)
        atomic_json(root/'completion.json',dict(completed_at=now(),planned_models=len(selected)*6,technical=technical))
    return root
