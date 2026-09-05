"""Record a reviewed code/environment freeze without rewriting scientific artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from .execution import atomic_json, git_state, now, runtime, source_hash
from .protocol import sha256_file
from .validate_preregistration import DEFAULT_CONFIG, DEFAULT_MANIFEST, DEFAULT_PREREG, REPO, validate


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,help='New execution-freeze JSON, outside protocol source files')
    args=parser.parse_args(argv)
    if errors:=validate():
        raise SystemExit('\n'.join(errors))
    state=git_state(require_clean=True)
    output=Path(args.output).resolve()
    if output.exists():
        raise SystemExit('refusing to overwrite an existing freeze record')
    if (REPO/'dl_moe').resolve() in output.parents or (REPO/'docs').resolve() in output.parents:
        raise SystemExit('freeze records must be outside scientific source directories')
    paths=[DEFAULT_CONFIG,DEFAULT_MANIFEST,DEFAULT_PREREG,REPO/'docs/preregistration/DL_MOE_01_PREREGISTRATION.md']
    atomic_json(output,dict(created_at=now(),git=state,source_hash=source_hash(),environment=runtime(),
        scientific_artifacts={str(p.relative_to(REPO)):sha256_file(p) for p in paths}))
    print(f'wrote execution freeze record: {output}')


if __name__=='__main__':
    main()
