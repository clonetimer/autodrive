"""File-integrity tests only: synthetic ZIP fixtures are NOT simulator runs."""
import json
from pathlib import Path
import zipfile
import pytest
from src.utils.metadrive_assets import (asset_inventory, stage_assets, audit_upload,
                                        verify_staged_assets, REQUIRED_ASSETS)
from src.utils.installation import validate_target_base


def archives(tmp_path):
    source, assets = tmp_path / 'source.zip', tmp_path / 'assets.zip'
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('source/metadrive/version.py', "VERSION = '0.4.3'\n")
        z.writestr('source/setup.py', 'raise RuntimeError("must not execute while staging")')
    with zipfile.ZipFile(assets, 'w') as z:
        for name in REQUIRED_ASSETS:
            z.writestr('assets/' + name, '0.4.3' if name == 'version.txt' else 'asset fixture')
        z.writestr('assets/models/test.gltf', json.dumps({'buffers': [{'uri': 'test.bin'}]}))
        z.writestr('assets/models/test.bin', b'fixture')
    return source, assets


def test_upload_installer_name_is_not_dependency_evidence(tmp_path):
    p = tmp_path / 'MetaDrive_install.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('INSTALL.md', 'pip install metadrive-simulator')
        z.writestr('install.sh', 'echo demo')
    report = audit_upload(p)
    assert report['documentation_only'] and report['wheel_count'] == 0
    assert not report['runtime_verified']


def test_full_asset_inventory_checks_gltf_references(tmp_path):
    _, assets = archives(tmp_path)
    report = asset_inventory(assets, '0.4.3')
    assert report['file_count'] == 5
    assert report['checked_gltf_references'] == 1
    assert report['matches_source']


def test_version_file_alone_is_not_a_complete_asset_release(tmp_path):
    path = tmp_path / 'incomplete.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('assets/version.txt', '0.4.3')
    with pytest.raises(ValueError, match='Incomplete'):
        asset_inventory(path, '0.4.3')


def test_missing_gltf_buffer_is_reported(tmp_path):
    _, path = archives(tmp_path)
    with zipfile.ZipFile(path, 'a') as z:
        z.writestr('assets/models/broken.gltf', '{"buffers":[{"uri":"missing.bin"}]}')
    with pytest.raises(ValueError, match='Unresolved'):
        asset_inventory(path, '0.4.3')


def test_remote_asset_reference_is_not_downloaded(tmp_path):
    _, path = archives(tmp_path)
    with zipfile.ZipFile(path, 'a') as z:
        z.writestr('assets/models/remote.gltf', '{"images":[{"uri":"https://example.com/a.png"}]}')
    with pytest.raises(ValueError, match='Remote'):
        asset_inventory(path, '0.4.3')


def test_stage_is_atomic_offline_and_idempotent(tmp_path):
    source, assets = archives(tmp_path)
    report = stage_assets(source, assets, tmp_path / 'external')
    assert report['real_steps_executed'] == 0 and not report['pip_executed']
    assert Path(report['source_directory'], 'metadrive/assets/version.txt').read_text() == '0.4.3'
    assert stage_assets(source, assets, tmp_path / 'external')['status'] == 'already_staged_and_reverified'


def test_tampering_with_staged_asset_prevents_reuse(tmp_path):
    source, assets = archives(tmp_path)
    report = stage_assets(source, assets, tmp_path / 'external')
    Path(report['source_directory'], 'metadrive/assets/models/test.bin').write_text('changed')
    with pytest.raises(ValueError, match='verification failed'):
        stage_assets(source, assets, tmp_path / 'external')


def test_tampering_with_staged_source_prevents_reuse(tmp_path):
    source, assets = archives(tmp_path)
    report = stage_assets(source, assets, tmp_path / 'external')
    Path(report['source_directory'], 'setup.py').write_text('changed')
    with pytest.raises(ValueError, match='source was modified'):
        stage_assets(source, assets, tmp_path / 'external')


def test_existing_unmanaged_source_is_protected(tmp_path):
    source, assets = archives(tmp_path)
    (tmp_path / 'external/metadrive').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='without a staging receipt'):
        stage_assets(source, assets, tmp_path / 'external')


def test_staging_unsafe_source_never_escapes(tmp_path):
    source, assets = archives(tmp_path)
    with zipfile.ZipFile(source, 'a') as z:
        z.writestr('../escaped', 'x')
    with pytest.raises(ValueError, match='Unsafe'):
        stage_assets(source, assets, tmp_path / 'external')
    assert not (tmp_path / 'escaped').exists()


def test_delta_baseline_requires_exact_versions():
    import importlib.metadata
    version = importlib.metadata.version('numpy')
    assert validate_target_base({'numpy': version})['status'] == 'verified'
    with pytest.raises(ValueError, match='baseline mismatch'):
        validate_target_base({'numpy': '0.0.0'})
    with pytest.raises(ValueError, match='baseline mismatch'):
        validate_target_base({'definitely-not-installed-package': '1.0'})


def test_project_dependency_graph_reports_actual_missing_and_versions(monkeypatch):
    from types import SimpleNamespace
    import importlib.metadata
    from src.utils.installation import check_installed_closure
    dists = {
        'root': SimpleNamespace(version='1.0', metadata={}, requires=['child>=2', 'absent']),
        'child': SimpleNamespace(version='1.0', metadata={}, requires=[]),
    }
    def distribution(name):
        if name not in dists:
            raise importlib.metadata.PackageNotFoundError(name)
        return dists[name]
    monkeypatch.setattr('src.utils.installation.importlib.metadata.distribution', distribution)
    report = check_installed_closure(['root'])
    assert report['status'] == 'failed'
    assert {r['reason'] for r in report['issues']} == {'missing', 'version_mismatch'}
    dists['child'].version = '2.0'
    dists['root'].requires = ['child>=2']
    assert check_installed_closure(['root'])['status'] == 'passed'
