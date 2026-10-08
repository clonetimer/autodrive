"""Build a single-root source+evidence archive without vendor binaries/caches.

Run final tests before this command. This packages actual files and hashes them;
it does not upgrade blocked acceptance gates into a successful stable release.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import zipfile

EXCLUDED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'external',
                  'coverage_html', 'dev_idm_eval', 'dev_ppo_eval', 'build', 'dist'}
EXCLUDED_NAMES = {'.coverage', 'MANIFEST.sha256', 'training_process.json',
                  'evaluation_process.json', 'curriculum_process.json'}


def files_to_ship(root: Path):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in EXCLUDED_PARTS or part.startswith('dev_smoke') or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix in {'.pyc', '.tmp', '.part'}:
            continue
        # SB3 stores policies as ZIP archives. Ship only curated model ZIPs under models/;
        # other ZIPs are dependency/upload/checkpoint material and remain excluded.
        if path.suffix == '.zip' and not (relative.parts and relative.parts[0] == 'models'):
            continue
        # Never redistribute font files or binary dependency packages.
        if path.suffix.lower() in {'.ttf', '.otf', '.woff', '.woff2', '.whl'}:
            continue
        yield path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def package(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    (root / 'docs').mkdir(exist_ok=True)
    tree = root / 'docs/TREE.txt'
    tree.touch(exist_ok=True)
    paths = [str(p.relative_to(root)) for p in files_to_ship(root)] + ['MANIFEST.sha256']
    tree.write_text('rl_autodrive_agent/\n' + '\n'.join('  ' + p for p in sorted(paths)) + '\n', encoding='utf-8')
    files = list(files_to_ship(root))
    manifest = root / 'MANIFEST.sha256'
    manifest.write_text(''.join(f'{sha256(path)}  {path.relative_to(root).as_posix()}\n' for path in files), encoding='utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in [*files, manifest]:
            bundle.write(path, 'rl_autodrive_agent/' + path.relative_to(root).as_posix())
    temp.replace(output)
    digest = sha256(output)
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n')
    return {'archive': str(output), 'sha256': digest, 'files': len(files) + 1,
            'bytes': output.stat().st_size, 'root': 'rl_autodrive_agent',
            'status': 'metadrive_research_release_not_safety_certified'}


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(package(Path(args.root), Path(args.output)), indent=2))
