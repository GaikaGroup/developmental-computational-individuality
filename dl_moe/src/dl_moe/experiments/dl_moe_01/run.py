from __future__ import annotations

import argparse
import copy

from ...training import load_config
from .execution import execute
from .protocol import parameter_audit, sha256_file
from .validate_preregistration import DEFAULT_CONFIG, DEFAULT_MANIFEST, DEFAULT_PREREG, validate


def technical_config(config):
    config=copy.deepcopy(config)
    config['model'].update(model_dim=8,expert_hidden_dim=12,experts_per_population=2,comm_dim=2)
    config['training'].update(total_steps=8,early_steps=4,batch_size=16)
    config['evaluation'].update(checkpoints=[4,6,8],examples=40)
    config['device']='cpu'
    return config


def main(argv=None):
    p=argparse.ArgumentParser(description='DL-MoE-01 execution; no training by default')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--preregistered',action='store_true'); modes.add_argument('--technical',action='store_true')
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--config',default=str(DEFAULT_CONFIG)); p.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    p.add_argument('--preregistration',default=str(DEFAULT_PREREG)); p.add_argument('--output')
    a=p.parse_args(argv)
    if errors:=validate(a.config,a.manifest,a.preregistration):
        raise SystemExit('\n'.join(errors))
    config=load_config(a.config); audit=parameter_audit(config)
    if a.dry_run:
        print('DL-MoE-01 DRY RUN (read-only)\n30 primary seed blocks; 6 models per block; 180 planned models; reserves excluded')
        print(f"DL params: {audit['dl_params']}; Flat params: {audit['flat_params']}; ratio: {audit['ratio']:.6f}")
        print('Primary endpoint: architecture_amplification_a (A_s), step 20000, renormalized expert knockout')
        print(f'seed manifest SHA256: {sha256_file(a.manifest)}')
        return
    if not (a.preregistered or a.technical):
        p.error('select --dry-run, --technical, or explicitly --preregistered')
    if not a.output:
        p.error('--output is required for execution')
    execute(technical_config(config) if a.technical else config,a.output,technical=a.technical,resume=a.resume)


if __name__=='__main__':
    main()
