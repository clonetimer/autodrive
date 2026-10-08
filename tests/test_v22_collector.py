"""Offline resolver/metadata tests. No downloaded fixture is called a real wheel."""
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

SPEC = importlib.util.spec_from_file_location('collector', Path(__file__).parents[1] / 'tools/collect_metadrive_wheels.py')
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def target(roots, installed=None):
    return {'schema': 1, 'fingerprint': 'fixture', 'root_requirements': roots,
            'python_version': '3.13', 'implementation': 'cp', 'abis': ['cp313', 'abi3', 'none'],
            'platforms': ['manylinux_2_17_x86_64'],
            'compatible_tags': ['cp313-cp313-manylinux_2_17_x86_64', 'py3-none-any'],
            'marker_environment': {'sys_platform': 'linux', 'python_version': '3.13',
                'python_full_version': '3.13.5', 'os_name': 'posix', 'platform_machine': 'x86_64',
                'platform_system': 'Linux', 'implementation_name': 'cpython'},
            'installed': installed or {}}


def record(name, deps=(), version='1.0', tags=None):
    return {'name': name, 'version': version, 'requires_dist': list(deps),
            'requires_python': '>=3.10', 'tags': tags or ['py3-none-any'], 'file': f'{name}.whl'}


def test_target_linux_markers_not_host_windows_markers():
    data = {'parent': record('parent', ['linuxdep; sys_platform == "linux"', 'windep; sys_platform == "win32"']),
            'linuxdep': record('linuxdep')}
    result = collector.resolve(target(['parent']), lambda req: data[req])
    assert {r['name'] for r in result['wheels']} == {'parent', 'linuxdep'}


def test_existing_target_packages_are_reused_not_downloaded():
    numpy = record('numpy', version='2.3.5')
    result = collector.resolve(target(['parent'], {'numpy': numpy}),
                               lambda req: record('parent', ['numpy>=2']))
    assert result['baseline'] == {'numpy': '2.3.5'}
    assert len(result['wheels']) == 1


def test_incompatible_frozen_base_is_not_silently_upgraded():
    with pytest.raises(ValueError, match='Frozen target conflict'):
        collector.resolve(target(['numpy<2'], {'numpy': record('numpy', version='2.3.5')}),
                          lambda req: pytest.fail('must not fetch replacement'))


def test_transitive_conflict_fails_closed():
    records = {'a': record('a', ['c>=1']), 'b': record('b', ['c<1']), 'c': record('c')}
    with pytest.raises(ValueError, match='conflict'):
        collector.resolve(target(['a', 'b']), lambda req: records[collector.Requirement(req).name])


def test_extras_are_propagated_and_dormant_extras_are_ignored():
    data = {'a': record('a', ['b; extra == "test"', 'c; extra == "unused"']), 'b': record('b')}
    result = collector.resolve(target(['a[test]']), lambda req: data[collector.Requirement(req).name])
    assert {r['name'] for r in result['wheels']} == {'a', 'b'}


def test_cycles_terminate():
    data = {'a': record('a', ['b']), 'b': record('b', ['a'])}
    assert len(collector.resolve(target(['a']), lambda req: data[req])['wheels']) == 2


def test_cp311_windows_wheel_cannot_satisfy_cp313_linux():
    with pytest.raises(ValueError, match='Invalid downloaded'):
        collector.resolve(target(['a']), lambda req: record('a', tags=['cp311-cp311-win_amd64']))


def test_requires_python_is_enforced_for_existing_base():
    rec = record('a'); rec['requires_python'] = '<3.13'
    with pytest.raises(ValueError, match='Python constraint'):
        collector.resolve(target(['a'], {'a': rec}), lambda req: pytest.fail())


def test_download_command_explicitly_targets_the_exported_abi(tmp_path):
    cmd = collector.download_command('panda3d>=1.10.15', target([]), tmp_path, 'https://pypi.org/simple')
    assert '--no-deps' in cmd and '--only-binary=:all:' in cmd
    assert cmd[cmd.index('--python-version') + 1] == '3.13'
    assert 'manylinux_2_17_x86_64' in cmd and 'cp313' in cmd


def test_direct_url_requirement_rejected():
    with pytest.raises(ValueError, match='Direct URL'):
        collector.resolve(target(['thing @ https://example.com/thing.whl']), lambda req: pytest.fail())


def test_read_wheel_checks_filename_metadata_agreement(tmp_path):
    wheel = tmp_path / 'example-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('example-1.0.dist-info/METADATA', 'Name: example\nVersion: 2.0\n')
    with pytest.raises(ValueError, match='disagree'):
        collector.read_wheel(wheel)


def test_plan_does_not_fetch_anything():
    result = collector.plan(target(['a', 'b'], {'a': record('a')}))
    assert result['missing_roots'] == ['b']
    assert result['mode'] == 'plan_only_no_download'


def test_cpu_collector_refuses_cuda_dependencies():
    with pytest.raises(ValueError, match='CPU-only'):
        collector.resolve(target(['nvidia-cublas-cu12']), lambda req: pytest.fail('must not download'))
