from pathlib import Path
import json
import zipfile
import pytest
from src.utils.installation import (inspect_source_archive, inspect_assets_archive,
    build_wheel_manifest, validate_wheelhouse, network_diagnostics)
from src.utils.logger import sha256


def source_zip(path, version="'0.4.3'"):
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr('metadrive/metadrive/version.py', f'VERSION = {version}\n')
        bundle.writestr('metadrive/setup.py', 'raise RuntimeError("must not execute")')


def wheel(path):
    # Deliberately minimal metadata fixture, not a runtime wheel or install test.
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr('example-1.0.dist-info/METADATA', 'Name: example\nVersion: 1.0\nRequires-Dist: numpy>=1\n')


def test_source_inspection_without_execution_and_sha(tmp_path):
    path = tmp_path / 'source.zip'
    source_zip(path)
    result = inspect_source_archive(path, sha256(path))
    assert result['source_version'] == '0.4.3'
    assert not result['assets_in_source']
    with pytest.raises(ValueError, match='SHA-256'):
        inspect_source_archive(path, '0' * 64)


def test_source_version_code_cannot_execute(tmp_path):
    path = tmp_path / 'unsafe.zip'
    marker = tmp_path / 'executed'
    source_zip(path, f"__import__('pathlib').Path({str(marker)!r}).touch()")
    with pytest.raises(ValueError):
        inspect_source_archive(path)
    assert not marker.exists()


def test_assets_require_matching_version_and_root(tmp_path):
    path = tmp_path / 'assets.zip'
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr('assets/version.txt', '0.4.3\n')
    assert inspect_assets_archive(path, '0.4.3')['matches_source']
    with pytest.raises(ValueError, match='version mismatch'):
        inspect_assets_archive(path, '0.4.2')
    with pytest.raises(ValueError, match='SHA-256'):
        inspect_assets_archive(path, '0.4.3', '0' * 64)


def test_wheel_checksum_tag_and_unlisted_file_gate(tmp_path):
    path = tmp_path / 'example-1.0-py3-none-any.whl'
    wheel(path)
    manifest_path = tmp_path / 'manifest.json'
    manifest = build_wheel_manifest(tmp_path, manifest_path)
    assert manifest['wheels'][0]['requires_dist'] == ['numpy>=1']
    assert validate_wheelhouse(tmp_path, manifest_path)['wheels'] == 1
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        validate_wheelhouse(tmp_path, manifest_path)
    build_wheel_manifest(tmp_path, manifest_path)
    wheel(tmp_path / 'extra-1.0-py3-none-any.whl')
    with pytest.raises(ValueError, match='Unlisted'):
        validate_wheelhouse(tmp_path, manifest_path)


def test_incompatible_platform_rejected_before_install(tmp_path):
    wheel(tmp_path / 'example-1.0-cp39-cp39-win_amd64.whl')
    manifest_path = tmp_path / 'manifest.json'
    build_wheel_manifest(tmp_path, manifest_path)
    with pytest.raises(ValueError, match='ABI/platform'):
        validate_wheelhouse(tmp_path, manifest_path)


def test_network_dns_failure_does_not_retry_pip_or_https(monkeypatch):
    from types import SimpleNamespace
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr='Temporary failure in name resolution')
    monkeypatch.setattr('src.utils.installation.subprocess.run', run)
    result = network_diagnostics()
    assert result['https']['status'] == 'not_attempted_due_to_dns_failure'
    assert len(commands) == 3
    assert all('pip' not in command for command in commands)


def test_manifest_rejects_path_escape(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema': 1, 'wheels': [{'file': '../outside.whl'}]}))
    with pytest.raises(ValueError, match='Unsafe'):
        validate_wheelhouse(tmp_path, path)


def test_prerequisite_does_not_confuse_assets_presence_with_matching_version(tmp_path, monkeypatch):
    from src.envs.metadrive_wrapper import prerequisite_status
    package = tmp_path / 'metadrive'
    (package / 'assets').mkdir(parents=True)
    (package / 'version.py').write_text("VERSION = '0.4.3'\n")
    (package / 'assets/version.txt').write_text('0.4.2')
    monkeypatch.setattr('src.envs.metadrive_wrapper.importlib.util.find_spec', lambda name: object())
    status = prerequisite_status(str(tmp_path))
    assert status['assets_present'] and not status['ready']
    assert not status['assets_version_matches']
    (package / 'assets/version.txt').write_text('0.4.3')
    assert prerequisite_status(str(tmp_path))['ready']
