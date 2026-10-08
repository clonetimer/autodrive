"""Auditable network/source/offline dependency checks without simulator emulation.

A wheelhouse manifest detects accidental modification, not malicious replacement
of both manifest and wheel: it is deliberately not called a signature.
"""
from __future__ import annotations
import ast
import email
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import zipfile
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from src.utils.logger import sha256, save_json


def network_diagnostics(hosts=('pypi.org', 'files.pythonhosted.org', 'github.com'), timeout=6) -> dict:
    """Run each DNS probe in a timed subprocess: resolver calls can otherwise hang."""
    result = {}
    for host in hosts:
        if not isinstance(host, str) or any(c in host for c in '\r\n /'):
            raise ValueError('Expected a DNS hostname')
        command = [sys.executable, '-c', 'import socket,sys,json; print(json.dumps(socket.getaddrinfo(sys.argv[1],443)))', host]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
            result[host] = {'dns': 'passed' if run.returncode == 0 else 'failed',
                            'detail': (run.stdout or run.stderr)[-1500:]}
        except subprocess.TimeoutExpired:
            result[host] = {'dns': 'timeout', 'detail': f'Exceeded {timeout}s'}
    result['https'] = {'status': 'not_attempted_due_to_dns_failure'}
    if result.get('pypi.org', {}).get('dns') == 'passed':
        command = [sys.executable, '-c',
                   'import urllib.request; r=urllib.request.urlopen("https://pypi.org/simple/gymnasium/",timeout=8); print(r.status)']
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            result['https'] = {'status': 'passed' if run.returncode == 0 else 'failed',
                               'detail': (run.stdout or run.stderr)[-1500:]}
        except subprocess.TimeoutExpired:
            result['https'] = {'status': 'timeout'}
    return result


def inspect_source_archive(path: str | Path, expected_sha256: str | None = None) -> dict:
    path = Path(path)
    digest = sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('Source archive SHA-256 mismatch')
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        candidates = [name for name in names if name.endswith('/metadrive/version.py')]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one MetaDrive source root')
        # Never execute version.py/setup.py during inspection.
        text = bundle.read(candidates[0]).decode('utf-8')
        version = None
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in node.targets):
                version = ast.literal_eval(node.value)
        if not isinstance(version, str):
            raise ValueError('Source VERSION must be a string literal')
        prefix = candidates[0].removesuffix('metadrive/version.py')
        asset_name = prefix + 'metadrive/assets/version.txt'
        assets_version = bundle.read(asset_name).decode().strip() if asset_name in names else None
        return {'path': str(path.resolve()), 'sha256': digest, 'source_version': version,
                'source_prefix': prefix, 'assets_in_source': asset_name in names, 'assets_version': assets_version}


def inspect_assets_archive(path: str | Path, source_version: str, expected_sha256=None) -> dict:
    path = Path(path)
    digest = sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('Assets archive SHA-256 mismatch')
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        if 'assets/version.txt' not in names:
            raise ValueError('Assets zip must contain assets/version.txt at its root')
        version = bundle.read('assets/version.txt').decode('utf-8').strip()
    if version != source_version:
        raise ValueError(f'Assets/source version mismatch: {version} != {source_version}')
    return {'sha256': digest, 'version': version, 'matches_source': True}


def build_wheel_manifest(directory: str | Path, output: str | Path | None = None) -> dict:
    directory = Path(directory)
    wheels = []
    for path in sorted(directory.glob('*.whl')):
        if path.is_symlink():
            raise ValueError('Wheel symlinks are not allowed')
        _, _, _, tags = parse_wheel_filename(path.name)
        with zipfile.ZipFile(path) as bundle:
            metadata = [name for name in bundle.namelist() if name.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                raise ValueError('Wheel has invalid METADATA count')
            record = email.message_from_bytes(bundle.read(metadata[0]))
        wheels.append({'file': path.name, 'sha256': sha256(path), 'tags': sorted(map(str, tags)),
                       'name': record.get('Name'), 'version': record.get('Version'),
                       'requires_dist': record.get_all('Requires-Dist', [])})
    if not wheels:
        raise ValueError('Wheelhouse is empty')
    manifest = {'schema': 1, 'builder': {'python': platform.python_version(), 'platform': sysconfig.get_platform(),
                 'machine': platform.machine(), 'soabi': sysconfig.get_config_var('SOABI')},
                'wheels': wheels, 'dependency_closure': 'must_be_verified_by_pip_resolver_and_pip_check'}
    if output:
        save_json(manifest, output)
    return manifest


def validate_wheelhouse(directory: str | Path, manifest_path: str | Path) -> dict:
    directory = Path(directory)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema') != 1 or not manifest.get('wheels'):
        raise ValueError('Invalid or empty wheelhouse manifest')
    compatible = set(sys_tags())
    seen = set()
    for row in manifest['wheels']:
        name = row['file']
        if Path(name).name != name or name in seen:
            raise ValueError('Unsafe or duplicate wheel manifest entry')
        seen.add(name)
        path = directory / name
        if not path.is_file() or path.is_symlink() or sha256(path) != row['sha256']:
            raise ValueError(f'Wheel checksum/missing file: {name}')
        _, _, _, tags = parse_wheel_filename(name)
        if not (tags & compatible):
            raise ValueError(f'Incompatible wheel ABI/platform for this interpreter: {name}')
        if sorted(map(str, tags)) != row['tags']:
            raise ValueError('Wheel tags disagree with manifest')
    if seen != {path.name for path in directory.glob('*.whl')}:
        raise ValueError('Unlisted wheels in wheelhouse; rebuild the manifest')
    baseline = validate_target_base(manifest.get('requires_target_base', {}))
    return {'status': 'verified_files_and_tags', 'wheels': len(seen), 'baseline': baseline,
            'note': 'Dependency closure is checked separately by pip; this is not a digital signature.'}


def validate_target_base(required: dict) -> dict:
    """Delta wheels are valid only on the exact baseline that the collector reused."""
    if not isinstance(required, dict):
        raise ValueError('Target baseline must be a name/version mapping')
    mismatches = []
    for name, expected in sorted(required.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            mismatches.append({'name': name, 'expected': expected, 'actual': actual})
    if mismatches:
        raise ValueError(f'Delta wheelhouse baseline mismatch: {mismatches[:8]}. Do not install on a different base.')
    return {'status': 'verified' if required else 'not_required', 'packages': len(required)}


def check_installed_closure(roots: list[str]) -> dict:
    """Check a project's installed dependency graph, separately from global pip check.

    Does not mark a globally conflicted overlay as clean. An unrelated inherited
    package must never be silently conflated with a missing simulator dependency.
    """
    from collections import deque
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    from packaging.utils import canonicalize_name
    from packaging.specifiers import SpecifierSet
    environment = default_environment()
    queue = deque((text, 'root', ()) for text in roots)
    processed, extras, versions, issues = {}, {}, {}, []
    while queue:
        text, parent, parent_extras = queue.popleft()
        requirement = Requirement(text)
        if requirement.url:
            issues.append({'parent': parent, 'requirement': text, 'reason': 'unsupported_direct_url'})
            continue
        if requirement.marker and not any(requirement.marker.evaluate({**environment, 'extra': extra})
                                          for extra in {'', *parent_extras}):
            continue
        name = canonicalize_name(requirement.name)
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            issues.append({'parent': parent, 'requirement': text, 'reason': 'missing'})
            continue
        versions[name] = dist.version
        if not requirement.specifier.contains(dist.version, prereleases=True):
            issues.append({'parent': parent, 'requirement': text, 'actual': dist.version, 'reason': 'version_mismatch'})
        python = dist.metadata.get('Requires-Python', '')
        if python and not SpecifierSet(python).contains(environment['python_full_version']):
            issues.append({'parent': parent, 'requirement': text, 'reason': 'python_mismatch'})
        extras.setdefault(name, set()).update(requirement.extras)
        state = tuple(sorted(extras[name]))
        if processed.get(name) != state:
            processed[name] = state
            queue.extend((dep, name, state) for dep in (dist.requires or []))
    return {'status': 'passed' if not issues else 'failed', 'roots': roots,
            'installed_versions': versions, 'issues': issues,
            'scope': 'requested_distribution_graph_only_not_global_environment'}
