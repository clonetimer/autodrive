#!/usr/bin/env python3
"""Collect REAL wheels for an exported target, even from Windows/macOS.

Run on a NETWORKED computer. No simulator/model download, no CUDA download,
no host package installation. Default is an additive bundle for the recorded
sandbox base. Reused packages must still exist at exactly the recorded versions.
The conservative resolver fails on conflicts rather than claiming a false lock.
"""
from __future__ import annotations

import argparse
from collections import deque
import email
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import parse_tag, sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
except ImportError:
    # A normal pip installation already ships this; no extra host setup needed.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.tags import parse_tag, sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def read_wheel(path: Path) -> dict:
    name, version, _, tags = parse_wheel_filename(path.name)
    with zipfile.ZipFile(path) as archive:
        metas = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metas) != 1:
            raise ValueError(f'Invalid wheel metadata: {path.name}')
        meta = email.message_from_bytes(archive.read(metas[0]))
    if canonicalize_name(meta['Name']) != name or str(version) != meta['Version']:
        raise ValueError(f'Wheel filename and metadata disagree: {path.name}')
    return {'file': path.name, 'name': name, 'version': str(version),
            'sha256': digest(path), 'tags': sorted(map(str, tags)),
            'requires_dist': meta.get_all('Requires-Dist', []),
            'requires_python': meta.get('Requires-Python', '')}


def eligible(requirement: Requirement, target: dict, extras=()) -> bool:
    if requirement.url:
        raise ValueError('Direct URL requirements are not supported by this offline collector')
    return requirement.marker is None or any(requirement.marker.evaluate(
        {**target['marker_environment'], 'extra': extra}) for extra in {'', *extras})


def compatible(record: dict, target: dict) -> bool:
    target_tags = {tag for text in target['compatible_tags'] for tag in parse_tag(text)}
    wheel_tags = {tag for text in record['tags'] for tag in parse_tag(text)}
    return bool(target_tags & wheel_tags) and (not record.get('requires_python') or
        SpecifierSet(record['requires_python']).contains(target['marker_environment']['python_full_version']))


def download_command(requirement: str, target: dict, directory: Path, index_url: str) -> list[str]:
    cmd = [sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check',
           '--no-deps', '--only-binary=:all:', '--dest', str(directory),
           '--index-url', index_url, '--implementation', target['implementation'],
           '--python-version', target['python_version'], '--retries', '1', '--timeout', '30']
    for abi in target['abis']:
        cmd += ['--abi', abi]
    for name in target['platforms']:
        cmd += ['--platform', name]
    return cmd + [requirement]


class Downloader:
    def __init__(self, target, directory, index_url):
        self.target, self.directory, self.index_url = target, directory, index_url
        self.commands = []

    def run(self, command, timeout=900):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        self.commands.append({'command': command, 'exit_code': result.returncode,
                              'output': result.stdout})
        if result.returncode:
            raise RuntimeError(f'Command failed:\n{result.stdout[-6000:]}')

    def __call__(self, requirement: str) -> dict:
        with tempfile.TemporaryDirectory(prefix='md-wheels-') as tmp:
            tmp = Path(tmp)
            name = canonicalize_name(Requirement(requirement).name)
            if name == 'progressbar':
                # PyPI progressbar 2.5 only has an sdist. Build its real wheel on
                # the collector host; accept it only if it is platform-independent.
                self.run([sys.executable, '-m', 'pip', 'wheel', '--disable-pip-version-check',
                          '--no-deps', '--wheel-dir', str(tmp), '--index-url', self.index_url,
                          '--retries', '1', '--timeout', '30', requirement])
            else:
                self.run(download_command(requirement, self.target, tmp, self.index_url))
            wheels = list(tmp.glob('*.whl'))
            if len(wheels) != 1:
                raise ValueError(f'Expected one actual wheel for {requirement}')
            record = read_wheel(wheels[0])
            if not compatible(record, self.target):
                raise ValueError(f'Incompatible wheel selected: {record["file"]}')
            if name == 'progressbar' and not all(tag.endswith('-none-any') for tag in record['tags']):
                raise ValueError('Host-built progressbar wheel is not pure Python; refused')
            import shutil
            shutil.copyfile(wheels[0], self.directory / wheels[0].name)
            return record


def resolve(target: dict, fetch, full: bool = False) -> dict:
    """Target-marker dependency traversal with a frozen installed baseline.

    Resolves ordinary wheels conservatively; no speculative backtracking or
    source emulation. Test fetchers exercise this graph, not real installation.
    """
    if target.get('schema') != 1:
        raise ValueError('Unsupported target schema')
    installed = {} if full else target['installed']
    constraints, extras, selected, processed = {}, {}, {}, {}
    queue = deque((text, 'root', ()) for text in target['root_requirements'])
    edges = []
    while queue:
        text, parent, parent_extras = queue.popleft()
        req = Requirement(text)
        if not eligible(req, target, parent_extras):
            continue
        name = canonicalize_name(req.name)
        if name.startswith('nvidia-'):
            raise ValueError('CPU-only dependency collector refuses CUDA dependency downloads')
        constraints.setdefault(name, set()).add(str(req.specifier))
        extras.setdefault(name, set()).update(req.extras)
        edges.append({'parent': parent, 'requirement': str(req)})
        specifier = SpecifierSet(','.join(sorted(s for s in constraints[name] if s)))
        if name not in selected:
            if name in installed:
                rec = dict(installed[name], source='existing_target_base')
                if not specifier.contains(rec['version'], prereleases=True):
                    raise ValueError(f'Frozen target conflict: {name}=={rec["version"]} does not satisfy {specifier}. '
                                     'Export a new target; nothing was silently upgraded.')
            else:
                rec = dict(fetch(name + str(specifier)), source='bundled_wheel')
                if canonicalize_name(rec['name']) != name or not compatible(rec, target):
                    raise ValueError(f'Invalid downloaded dependency: {name}')
            selected[name] = rec
        rec = selected[name]
        if not specifier.contains(rec['version'], prereleases=True):
            raise ValueError(f'Dependency conflict for {name}: selected {rec["version"]}, required {specifier}. '
                             'No valid complete bundle was produced.')
        if rec.get('requires_python') and not SpecifierSet(rec['requires_python']).contains(
                target['marker_environment']['python_full_version']):
            raise ValueError(f'Python constraint not met: {name} {rec["requires_python"]}')
        state = tuple(sorted(extras[name]))
        if processed.get(name) != state:
            processed[name] = state
            for dep in rec.get('requires_dist', []):
                queue.append((dep, name, state))
    return {'status': 'metadata_closure_verified_runtime_not_tested',
            'mode': 'full' if full else 'delta_for_frozen_target',
            'wheels': [selected[n] for n in sorted(selected) if selected[n]['source'] == 'bundled_wheel'],
            'baseline': {n: selected[n]['version'] for n in sorted(selected) if selected[n]['source'] == 'existing_target_base'},
            'edges': edges,
            'notes': ['Target markers and wheel tags checked, not collector-host markers.',
                      'Installation, dynamic-library linking and real simulator smoke remain target-side gates.']}


def plan(target, full=False):
    missing, existing = [], []
    for text in target['root_requirements']:
        req = Requirement(text)
        record = target['installed'].get(canonicalize_name(req.name))
        if not full and record and req.specifier.contains(record['version'], prereleases=True):
            existing.append({'requirement': text, 'version': record['version']})
        else:
            missing.append(text)
    return {'mode': 'plan_only_no_download', 'python': target['python_version'],
            'platforms': target['platforms'], 'missing_roots': missing,
            'reuse_roots': existing, 'transitive_dependencies': 'resolved on the networked collector',
            'source_only_build': 'progressbar==2.5 must be built into a genuine pure-Python wheel',
            'example_command': download_command('panda3d>=1.10.15,<1.11', target, Path('wheelhouse'), 'https://pypi.org/simple')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default=str(Path(__file__).with_name('target.json')))
    parser.add_argument('--output', default='metadrive_wheelhouse.zip')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--index-url', default='https://pypi.org/simple')
    args = parser.parse_args()
    if not args.index_url.startswith('https://'):
        raise ValueError('Use an HTTPS package index')
    target = json.loads(Path(args.target).read_text(encoding='utf-8'))
    if args.plan_only:
        print(json.dumps(plan(target), indent=2, ensure_ascii=False))
        return 0
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; choose a new path to preserve prior evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output.with_suffix('.failure.json')
    with tempfile.TemporaryDirectory(prefix='metadrive-dependency-bundle-') as tmp:
        wheelhouse = Path(tmp) / 'wheelhouse'
        wheelhouse.mkdir()
        fetch = Downloader(target, wheelhouse, args.index_url)
        try:
            resolution = resolve(target, fetch)
            manifest = {'schema': 1, 'builder': {'python': platform.python_version(), 'system': platform.system()},
                        'wheels': resolution['wheels'], 'requires_target_base': resolution['baseline'],
                        'target_fingerprint': target['fingerprint'],
                        'dependency_closure': resolution['status']}
            save(manifest, wheelhouse / 'manifest.json')
            save(resolution, wheelhouse / 'resolution.json')
            save(target, wheelhouse / 'target.json')
            save(fetch.commands, wheelhouse / 'download_commands.json')
            (wheelhouse / 'requirements-to-install.txt').write_text(''.join(
                f'{r["name"]}=={r["version"]}\n' for r in resolution['wheels']), encoding='utf-8')
            partial = output.with_suffix('.part')
            with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(wheelhouse.iterdir()):
                    archive.write(file, f'wheelhouse/{file.name}')
            partial.replace(output)
            output.with_suffix(output.suffix + '.sha256').write_text(f'{digest(output)}  {output.name}\n')
            failure_path.unlink(missing_ok=True)
            print(f'Created {output}: {len(resolution["wheels"])} actual wheels. Upload this ZIP, not this script.')
            return 0
        except Exception as exc:
            save({'status': 'failed_no_complete_bundle', 'error': str(exc),
                  'commands': fetch.commands, 'target_fingerprint': target['fingerprint']}, failure_path)
            print(f'Collection failed; details: {failure_path}', file=sys.stderr)
            return 2


if __name__ == '__main__':
    raise SystemExit(main())
