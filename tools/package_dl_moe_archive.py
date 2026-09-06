"""Package the completed DL-MoE-01 study; never modify experiment inputs."""
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dl_moe_01/archive_v1'
BASE = ROOT / 'results/dl_moe_01/confirmatory_v1'
COMMIT = 'e57710fcb7b2e15b448ae2438f041d2a98fd6929'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {}
    groups = [[] for _ in range(4)]
    raw = BASE / 'raw'
    original = json.loads((raw / 'raw_manifest.json').read_text())['files']
    for name, expected in original.items():
        path = (raw / name).resolve()
        if not path.is_relative_to(raw.resolve()):
            raise ValueError(name)
        if digest(path) != expected:
            raise ValueError(f'Original raw mismatch: {name}')
        group = 1 + (int(Path(name).parts[1]) - 1) // 10 if path.suffix == '.pt' else 0
        if group not in range(4):
            raise ValueError(name)
        groups[group].append(path)
    print(f'Original raw manifest verified: {len(original)} files', flush=True)
    groups[0].append(raw / 'raw_manifest.json')
    for folder in ['tables', 'analysis', 'validation']:
        groups[0].extend(p for p in (BASE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    groups[0].extend([BASE / 'analysis_manifest.json', BASE / 'canonical_analysis.log', ROOT / 'results/dl_moe_01/execution_freeze.json'])
    names = ['evidence', 'checkpoints-001-010', 'checkpoints-011-020', 'checkpoints-021-030']
    for label, paths in zip(names, groups):
        target = OUT / f'dl-moe-01-{label}.zip'
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(paths):
                name = path.relative_to(ROOT).as_posix()
                inventory[name] = digest(path)
                archive.write(path, name)
        if target.stat().st_size >= 2_000_000_000:
            raise ValueError(f'Asset too large: {target}')
        print(f'Packaged {target.name}: {target.stat().st_size} bytes', flush=True)
    source = OUT / 'dl-moe-01-frozen-source.zip'
    subprocess.run(['git', 'archive', '--format=zip', f'--output={source}', COMMIT], cwd=ROOT, check=True)
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            if not item.is_dir():
                inventory[item.filename] = hashlib.sha256(archive.read(item)).hexdigest()
    for path in (ROOT / 'docs/archive/dl_moe_01').glob('*.md'):
        shutil.copy2(path, OUT / path.name)
    for name in ['LICENSE', 'LICENSE-CONTENT.md']:
        shutil.copy2(ROOT / name, OUT / name)
    shutil.copy2(ROOT / 'tools/verify_archive.py', OUT / 'verify_archive.py')
    shutil.copy2(ROOT / 'results/dl_moe_01/manuscript/dl_moe_01_article_ru.docx', OUT / 'dl_moe_01_article_ru.docx')
    # Standalone cover documents stay beside the reconstructed repository.
    # README.md would overwrite the original source README, so verify it through
    # asset checksums, not the extracted source inventory.
    (OUT / 'ARCHIVE_MANIFEST.json').write_text(json.dumps({'source_commit': COMMIT, 'files': inventory}, indent=2) + '\n')
    for path in sorted(OUT.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                with archive.open(item) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != inventory[item.filename]:
                    raise ValueError(f'ZIP member mismatch: {item.filename}')
        print(f'ZIP member checksums verified: {path.name}', flush=True)
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'SHA256SUMS.txt'))
    print(f'COMPLETE: {len(inventory)} extracted files', flush=True)


if __name__ == '__main__':
    main()
