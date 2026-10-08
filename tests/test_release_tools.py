from pathlib import Path
import importlib.util
import hashlib
import zipfile
import pytest


def module_from_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_verification_detects_modifications(tmp_path):
    verifier = module_from_script('verify_package')
    (tmp_path / 'file.txt').write_text('real data')
    digest = hashlib.sha256(b'real data').hexdigest()
    (tmp_path / 'MANIFEST.sha256').write_text(f'{digest}  file.txt\n')
    assert verifier.verify(tmp_path)['ok']
    (tmp_path / 'file.txt').write_text('changed')
    assert not verifier.verify(tmp_path)['ok']


def test_manifest_rejects_escape(tmp_path):
    verifier = module_from_script('verify_package')
    (tmp_path / 'MANIFEST.sha256').write_text('0' * 64 + '  ../bad\n')
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)


def test_single_root_package_omits_caches_fonts_and_self(tmp_path):
    packager = module_from_script('package_release')
    verifier = module_from_script('verify_package')
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'code.py').write_text('x = 1\n')
    (root / 'font.ttf').write_text('not a font; exclusion fixture')
    (root / '__pycache__').mkdir()
    (root / '__pycache__/cache.pyc').write_bytes(b'cache')
    (root / 'models').mkdir()
    (root / 'models/model.zip').write_bytes(b'curated sb3 model')
    (root / 'experiments').mkdir()
    (root / 'experiments/checkpoint.zip').write_bytes(b'nonrelease checkpoint')
    archive = tmp_path / 'release.zip'
    result = packager.package(root, archive)
    assert result['status'] == 'metadrive_research_release_not_safety_certified'
    with zipfile.ZipFile(archive) as bundle:
        assert all(name.startswith('rl_autodrive_agent/') for name in bundle.namelist())
        assert not any('.ttf' in name or '.pyc' in name for name in bundle.namelist())
        assert 'rl_autodrive_agent/models/model.zip' in bundle.namelist()
        assert 'rl_autodrive_agent/experiments/checkpoint.zip' not in bundle.namelist()
        bundle.extractall(tmp_path / 'extracted')
    assert verifier.verify(tmp_path / 'extracted/rl_autodrive_agent')['ok']
