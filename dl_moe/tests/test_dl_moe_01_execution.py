import csv
import json
from pathlib import Path

import pytest
import torch

from dl_moe.experiments.dl_moe_01 import execution
from dl_moe.experiments.dl_moe_01.artifacts import extract,freeze_raw,verify_raw,aligned_distance
from dl_moe.experiments.dl_moe_01.analyze import analyze
from dl_moe.experiments.dl_moe_01.run import main,technical_config
from dl_moe.experiments.dl_moe_01.validate_preregistration import DEFAULT_CONFIG,DEFAULT_MANIFEST,DEFAULT_PREREG,validate
from dl_moe.training import load_config


@pytest.fixture(autouse=True)
def single_thread():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def config():
    return technical_config(load_config(DEFAULT_CONFIG))


def test_readonly_dryrun_and_empty_manifest(tmp_path):
    before=DEFAULT_MANIFEST.read_bytes()
    main(['--dry-run'])
    assert DEFAULT_MANIFEST.read_bytes()==before
    empty=tmp_path/'empty.csv'; empty.write_text('')
    assert validate(DEFAULT_CONFIG,empty,DEFAULT_PREREG)


def test_end_to_end_raw_freeze_and_tables(tmp_path,config):
    root=execution.execute(config,tmp_path/'raw',technical=True)
    records=[json.loads(p.read_text()) for p in root.glob('raw/*/*/run.json')]
    assert len(records)==12
    for block in ('001','002'):
        for arch in ('dl_moe','flat_moe'):
            assert len({r['manifest']['initial_state_sha256'] for r in records if r['manifest']['seed_block_id']==block and r['manifest']['architecture']==arch})==1
    frozen=freeze_raw(root)
    assert frozen['technical']
    output=extract(root,tmp_path/'tables')
    assert (output/'seed_manifest.csv').read_bytes()==DEFAULT_MANIFEST.read_bytes()
    with (output/'seed_effects.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and all(r['primary_endpoint_available']=='True' for r in rows)
    with (output/'behavior_metrics.csv').open() as f:
        assert len(list(csv.DictReader(f)))==72
    with pytest.raises(ValueError,match='technical'):
        analyze(output,tmp_path/'confirmatory_analysis',DEFAULT_PREREG)
    result=analyze(output,tmp_path/'technical_analysis',DEFAULT_PREREG,technical=True)
    assert result['technical'] and not result['strong_support']
    with pytest.raises(ValueError,match='frozen'):
        execution.execute(config,root,technical=True,resume=True)
    file=next(root.glob('raw/*/*/evaluation_*.json'))
    file.write_text(file.read_text()+' ')
    with pytest.raises(ValueError,match='integrity'):
        verify_raw(root)


def test_resume_matches_uninterrupted_training(tmp_path,config):
    root=tmp_path/'interrupted'
    with pytest.raises(InterruptedError):
        execution.execute(config,root,technical=True,blocks=1,stop_after=3)
    execution.execute(config,root,technical=True,blocks=1,resume=True)
    baseline=execution.execute(config,tmp_path/'baseline',technical=True,blocks=1)
    for path in root.glob('raw/*/*/step_8.pt'):
        a=torch.load(path,weights_only=True)['model']
        b=torch.load(baseline/path.relative_to(root),weights_only=True)['model']
        assert all(torch.equal(a[k],b[k]) for k in a)
    with (root/'exclusions_and_incidents.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['replacement_seed_used']=='False'


def test_numerical_failure_retained_without_replacement(tmp_path,config,monkeypatch):
    def fail(*args,**kwargs):
        raise FloatingPointError('injected NaN')
    monkeypatch.setattr(execution,'evaluate_checkpoint',fail)
    root=execution.execute(config,tmp_path/'failed',technical=True,blocks=1)
    with (root/'run_manifest.csv').open() as f:
        records=list(csv.DictReader(f))
    assert len(records)==6 and all(r['run_status']=='numerical_failure' for r in records)
    execution.execute(config,root,technical=True,blocks=1,resume=True)
    with (root/'exclusions_and_incidents.csv').open() as f:
        assert len(list(csv.DictReader(f)))==6
    freeze_raw(root)
    tables=extract(root,tmp_path/'tables')
    with (tables/'seed_effects.csv').open() as f:
        assert list(csv.DictReader(f))[0]['primary_endpoint_available']=='False'


def test_execution_rejects_config_drift_on_resume(tmp_path,config):
    with pytest.raises(InterruptedError):
        execution.execute(config,tmp_path/'run',technical=True,blocks=1,stop_after=3)
    config['training']['learning_rate']=.1
    with pytest.raises(ValueError,match='identity'):
        execution.execute(config,tmp_path/'run',technical=True,blocks=1,resume=True)


def test_assignment_is_invariant_under_allowed_permutations():
    import numpy as np
    a=np.arange(16).reshape(8,2).astype(float)
    assert aligned_distance(a,a[::-1])==0
    assert aligned_distance(a,a[[7,6,5,4,3,2,1,0]],4)==0


def test_resume_repairs_interrupted_checkpoint_archive(tmp_path, config, monkeypatch):
    root = tmp_path / 'checkpoint-interruption'
    original = execution.save_checkpoint
    failed = False

    def interrupt_archive(path, value):
        nonlocal failed
        if path.name == 'step_4.pt' and not failed:
            failed = True
            raise OSError('injected archive interruption')
        return original(path, value)

    monkeypatch.setattr(execution, 'save_checkpoint', interrupt_archive)
    with pytest.raises(OSError, match='archive interruption'):
        execution.execute(config, root, technical=True, blocks=1)
    execution.execute(config, root, technical=True, blocks=1, resume=True)
    freeze_raw(root)
    assert len(list(root.glob('raw/*/*/step_4.pt'))) == 6


def test_resume_repairs_interrupted_series_initialization(tmp_path, config, monkeypatch):
    root = tmp_path / 'initialization-interruption'
    original = execution.write_csv
    failed = False

    def interrupt_manifest(path, *args, **kwargs):
        nonlocal failed
        if path.name == 'protocol_hashes.csv' and not failed:
            failed = True
            raise OSError('injected initialization interruption')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(execution, 'write_csv', interrupt_manifest)
    with pytest.raises(OSError, match='initialization interruption'):
        execution.execute(config, root, technical=True, blocks=1)
    execution.execute(config, root, technical=True, blocks=1, resume=True)
    assert (root / 'protocol_hashes.csv').is_file()
    freeze_raw(root)
