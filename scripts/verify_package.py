"""Verify every release file against MANIFEST.sha256 (not a digital signature)."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = root / 'MANIFEST.sha256'
    if not manifest.is_file():
        raise FileNotFoundError('Missing release manifest')
    errors, paths = [], set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        checksum, relative = line.split('  ', 1)
        name = PurePosixPath(relative)
        if name.is_absolute() or '..' in name.parts or relative in paths:
            raise ValueError(f'Unsafe or duplicated manifest path: {relative}')
        paths.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f'missing/nonregular: {relative}')
        elif digest(path) != checksum:
            errors.append(f'checksum mismatch: {relative}')
    if not paths:
        raise ValueError('Empty manifest is not a verified release')
    return {'ok': not errors, 'checked_files': len(paths), 'errors': errors,
            'note': 'New user-generated files are allowed; manifest covers distributed release files only.'}


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    result = verify(Path(args.root))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['ok'] else 1)
