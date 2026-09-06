"""Verify extracted archive files without loading checkpoint objects."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'ARCHIVE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Unsafe archive path: {name}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    return len(manifest['files'])


if __name__ == '__main__':
    print(f'Verified {verify(Path.cwd())} files.')
