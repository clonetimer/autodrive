"""Stage the real, version-matched simulator assets without importing its engine.

Staging is independent of pip and DNS. It does NOT establish runtime readiness.
Every distributed asset is checked; a version.txt alone is insufficient.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from src.datasets.download import safe_extract_zip
from src.utils.installation import inspect_source_archive, inspect_assets_archive
from src.utils.logger import save_json, sha256

# This extra texture is checked by 0.4.3 AssetLoader.should_update_asset itself.
REQUIRED_ASSETS = (
    'version.txt', 'logo-tiny.png',
    'textures/grass1/GroundGrassGreen002_COL_1K.jpg',
)


def audit_upload(path: str | Path) -> dict:
    """Classify by contents, not a filename that says 'install' or 'dependencies'."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        files = [i for i in archive.infolist() if not i.is_dir()]
        names = [i.filename for i in files]
        wheels = [n for n in names if n.endswith('.whl')]
        return {'file': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                'files': len(files), 'uncompressed_bytes': sum(i.file_size for i in files),
                'wheel_files': wheels, 'wheel_count': len(wheels),
                'has_assets': 'assets/version.txt' in names,
                'has_source': any(n.endswith('/metadrive/version.py') for n in names),
                'documentation_only': all(Path(n).suffix.lower() in {'.md', '.pdf', '.html', '.sh', '.txt'} for n in names),
                'runtime_verified': False}


def asset_inventory(archive_path: str | Path, source_version: str) -> dict:
    """Validate CRC, file layout, glTF local references, and expected 0.4.3 texture."""
    report = inspect_assets_archive(archive_path, source_version)
    with zipfile.ZipFile(archive_path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 2 * 1024 ** 3:
            raise ValueError('Asset archive exceeds 2 GiB uncompressed limit')
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Corrupt ZIP member: {bad}')
        files = [i for i in archive.infolist() if not i.is_dir()]
        if len({i.filename for i in files}) != len(files):
            raise ValueError('Duplicate asset archive entries')
        names = {i.filename for i in files}
        if any(not n.startswith('assets/') for n in names):
            raise ValueError('Only assets/ entries may appear in the assets archive')
        missing = [n for n in REQUIRED_ASSETS if 'assets/' + n not in names]
        if missing:
            raise ValueError(f'Incomplete assets archive: {missing}')
        if not any(n.startswith('assets/models/') for n in names):
            raise ValueError('Missing models directory')
        references, missing_refs = 0, []
        for name in sorted(names):
            if not name.endswith('.gltf'):
                continue
            model = json.loads(archive.read(name))
            for entry in model.get('buffers', []) + model.get('images', []):
                uri = entry.get('uri', '')
                if not uri or uri.startswith('data:'):
                    continue
                # Normalize local relative references; never fetch remote resources.
                import posixpath
                from urllib.parse import unquote
                if '://' in uri:
                    raise ValueError(f'Remote glTF resource is not allowed: {name}')
                resolved = posixpath.normpath(posixpath.join(str(PurePosixPath(name).parent), unquote(uri)))
                references += 1
                if resolved not in names:
                    missing_refs.append({'model': name, 'resource': resolved})
        if missing_refs:
            raise ValueError(f'Unresolved glTF references: {missing_refs[:8]}')
        inventory = [{'path': i.filename.removeprefix('assets/'),
                      'bytes': i.file_size,
                      'sha256': __import__('hashlib').sha256(archive.read(i)).hexdigest()}
                     for i in sorted(files, key=lambda x: x.filename)]
        report.update(files=inventory, file_count=len(files),
                      bytes=sum(i.file_size for i in files), checked_gltf_references=references,
                      dataset_files=[n for n in sorted(names) if n.startswith('assets/waymo/') and n.endswith('.pkl')],
                      status='archive_integrity_verified_not_runtime')
    return report


def verify_staged_assets(root: str | Path, inventory: dict) -> dict:
    root = Path(root).resolve()
    errors = []
    for row in inventory['files']:
        p = root / row['path']
        if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file():
            errors.append({'file': row['path'], 'reason': 'missing_or_unsafe'})
        elif p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
            errors.append({'file': row['path'], 'reason': 'checksum'})
    if errors:
        raise ValueError(f'Staged asset verification failed: {errors[:8]}')
    return {'status': 'all_staged_files_verified', 'files': len(inventory['files'])}


def stage_assets(source_zip: str | Path, assets_zip: str | Path, workspace: str | Path) -> dict:
    """Atomic initial staging, idempotent revalidation; never replace edited sources."""
    source = inspect_source_archive(source_zip)
    assets = asset_inventory(assets_zip, source['source_version'])
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    destination = workspace / 'metadrive'
    receipt = workspace / 'metadrive_stage.json'
    if destination.exists():
        if not receipt.is_file():
            raise FileExistsError('Existing source without a staging receipt; use a new workspace (no overwrite)')
        previous = json.loads(receipt.read_text())
        if previous['source']['sha256'] != source['sha256'] or previous['assets']['sha256'] != assets['sha256']:
            raise FileExistsError('Existing source/assets differ; use a new workspace')
        verify_staged_assets(destination / 'metadrive/assets', assets)
        for row in previous['source_files']:
            path = destination / row['path']
            if path.is_symlink() or not path.is_file() or sha256(path) != row['sha256']:
                raise ValueError(f'Staged source was modified: {row["path"]}')
        return dict(previous, status='already_staged_and_reverified')
    with tempfile.TemporaryDirectory(prefix='.metadrive-stage-', dir=workspace) as tmp:
        tmp = Path(tmp)
        safe_extract_zip(source_zip, tmp / 'source', exclude_git=True)
        extracted = tmp / 'source' / source['source_prefix']
        if not (extracted / 'setup.py').is_file():
            raise ValueError('Source setup.py missing')
        existing_assets = extracted / 'metadrive/assets'
        if existing_assets.exists():
            shutil.rmtree(existing_assets)
        safe_extract_zip(assets_zip, extracted / 'metadrive')
        verify_staged_assets(existing_assets, assets)
        source_files = [{'path': str(p.relative_to(extracted)), 'sha256': sha256(p)}
                        for p in sorted(extracted.rglob('*')) if p.is_file()
                        and not p.is_relative_to(existing_assets)]
        extracted.rename(destination)
    report = {'status': 'source_and_assets_staged_runtime_not_installed',
              'source': source, 'assets': assets, 'source_files': source_files,
              'source_directory': str(destination), 'real_steps_executed': 0,
              'pip_executed': False}
    save_json(report, receipt)
    return report
