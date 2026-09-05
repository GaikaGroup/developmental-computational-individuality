from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ...training import load_config
from .protocol import MANIFEST_COLUMNS, parameter_audit, seed_rows, sha256_file

REPO = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = REPO/'dl_moe/configs/confirmatory/dl_moe_v1.yaml'
DEFAULT_MANIFEST = REPO/'dl_moe/experiments/dl_moe_01/seed_manifest.csv'
DEFAULT_PREREG = REPO/'dl_moe/experiments/dl_moe_01/preregistration.yaml'


def validate(config_path=DEFAULT_CONFIG, manifest_path=DEFAULT_MANIFEST, prereg_path=DEFAULT_PREREG):
    errors=[]
    try:
        config=load_config(config_path)
        audit=parameter_audit(config)
        if not audit['passes']:
            errors.append('parameter ratio exceeds 0.05')
        with Path(manifest_path).open(encoding='utf-8') as f:
            reader=csv.DictReader(f); columns=reader.fieldnames; rows=list(reader)
        if columns != list(MANIFEST_COLUMNS):
            errors.append('seed manifest columns mismatch')
        if rows != [{k:str(v) for k,v in row.items()} for row in seed_rows()]:
            errors.append('seed manifest values do not match deterministic generator')
        prereg=load_config(prereg_path)
        if prereg['experiment_id']!='DL-MoE-01' or prereg['sample_size']['planned_models']!=180:
            errors.append('unsupported experiment or sample size')
        # The candidate scientific artifacts remain byte-frozen. Code readiness is a separate check.
        overrides={'confirmatory_config':Path(config_path),'seed_manifest.csv':Path(manifest_path),
                   'preregistration.yaml':Path(prereg_path)}
        with (REPO/'dl_moe/experiments/dl_moe_01/protocol_hashes.csv').open() as f:
            hashes=list(csv.DictReader(f))
        for row in hashes:
            if row['artifact']=='code_commit':
                continue  # Historical preparation commit; runtime captures the actual execution commit.
            path=overrides.get(row['artifact'],REPO/row['path'])
            if sha256_file(path)!=row['sha256']:
                errors.append(f"scientific artifact hash mismatch: {row['artifact']}")
    except (OSError,KeyError,ValueError,TypeError) as exc:
        errors.append(f'invalid protocol input: {exc}')
    return errors


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(DEFAULT_CONFIG)); p.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    p.add_argument('--preregistration',default=str(DEFAULT_PREREG))
    a=p.parse_args(argv); errors=validate(a.config,a.manifest,a.preregistration)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Scientific artifacts validated; execution freeze still requires clean committed code and an environment snapshot.')


if __name__=='__main__':
    main()
