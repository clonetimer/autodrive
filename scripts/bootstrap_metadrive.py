"""Staged, fail-closed real MetaDrive installation with source/assets/wheel checks.

--preflight-only does not install or claim to run MetaDrive. Installing requires
an activated isolated environment unless --allow-current-environment is explicit.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import _bootstrap
from src.datasets.download import safe_extract_zip
from src.utils.installation import (network_diagnostics, inspect_source_archive,
    inspect_assets_archive, validate_wheelhouse, check_installed_closure)
from src.utils.logger import save_json, runtime_manifest
from src.utils.metadrive_assets import stage_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', required=True)
    parser.add_argument('--workspace', default='external')
    parser.add_argument('--wheelhouse')
    parser.add_argument('--wheelhouse-manifest')
    parser.add_argument('--assets-zip')
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--expected-assets-sha256')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--stage-only', action='store_true', help='Stage verified source/assets offline; does not install packages')
    parser.add_argument('--allow-current-environment', action='store_true')
    parser.add_argument('--allow-unrelated-base-conflicts', action='store_true',
                        help='Permit global pip-check conflicts ONLY when the requested installed graph passes; record the conflict')
    parser.add_argument('--report', default='reports/metadrive_installation.json')
    args = parser.parse_args()
    report = {'runtime': runtime_manifest(), 'stages': {}, 'real_steps_executed': 0,
              'mode': 'preflight_only' if args.preflight_only else 'install_and_smoke'}
    stage = 'source'
    try:
        report['stages'][stage] = inspect_source_archive(args.source_zip, args.expected_source_sha256)
        stage = 'assets'
        if args.assets_zip:
            report['stages'][stage] = inspect_assets_archive(args.assets_zip,
                report['stages']['source']['source_version'], args.expected_assets_sha256)
        else:
            report['stages'][stage] = {'status': 'requires_online_download'}
        if args.assets_zip and not args.preflight_only:
            stage = 'offline_staging'
            report['stages'][stage] = stage_assets(args.source_zip, args.assets_zip, args.workspace)
        if args.stage_only:
            if not args.assets_zip:
                raise ValueError('--stage-only requires --assets-zip')
            report['status'] = 'source_assets_staged_not_runtime_ready'
            return 0
        stage = 'dependencies'
        if args.wheelhouse:
            manifest = args.wheelhouse_manifest or str(Path(args.wheelhouse) / 'manifest.json')
            report['stages'][stage] = validate_wheelhouse(args.wheelhouse, manifest)
        if not args.wheelhouse or not args.assets_zip:
            stage = 'network'
            network = network_diagnostics()
            report['stages'][stage] = network
            if any(network.get(host, {}).get('dns') != 'passed' for host in ('pypi.org', 'files.pythonhosted.org', 'github.com')) or network['https']['status'] != 'passed':
                raise RuntimeError('Network preflight failed; no pip retries. Supply complete compatible wheelhouse AND matching assets zip.')
        if args.preflight_only:
            report['status'] = 'preflight_passed_runtime_not_tested'
            return 0
        stage = 'isolation'
        if sys.prefix == sys.base_prefix and not args.allow_current_environment:
            raise RuntimeError('Activate an isolated virtual environment before installation; global packages were not changed')
        workspace = Path(args.workspace).resolve()
        if args.assets_zip:
            source = Path(report['stages']['offline_staging']['source_directory'])
        else:
            safe_extract_zip(args.source_zip, workspace, exclude_git=True)
            source = workspace / report['stages']['source']['source_prefix']
        stage = 'pip_install'
        command = [sys.executable, '-m', 'pip', 'install']
        if args.wheelhouse:
            command += ['--no-index', '--find-links', str(Path(args.wheelhouse).resolve()), '--no-build-isolation']
        if args.wheelhouse:
            lock = Path(args.wheelhouse) / 'requirements-to-install.txt'
            if lock.is_file():
                command += ['-r', str(lock.resolve())]
        command += ['-e', str(source)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        report['stages'][stage] = {'command': command, 'exit_code': result.returncode,
                                  'stdout': result.stdout, 'stderr': result.stderr}
        if result.returncode:
            raise RuntimeError('pip installation failed; inspect recorded resolver/build errors')
        stage = 'assets_install'
        if args.assets_zip:
            # Already staged before dependency checks and reverified on repeat runs.
            report['stages'][stage] = {'status': 'verified_during_offline_staging'}
        else:
            subprocess.run([sys.executable, '-m', 'metadrive.pull_asset'], check=True, cwd=workspace.parent)
        stage = 'pip_check'
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        report['stages'][stage] = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        scoped = check_installed_closure(['metadrive-simulator==0.4.3', 'numpy>=1.26,<3', 'torch>=2.2,<3'])
        report['stages']['project_dependency_closure'] = scoped
        if scoped['status'] != 'passed':
            raise RuntimeError('Simulator/project dependency graph incomplete; do not proceed to training')
        if result.returncode and not args.allow_unrelated_base_conflicts:
            raise RuntimeError('Global pip check failed; explicit --allow-unrelated-base-conflicts is required even if project closure passes')
        if result.returncode:
            report['global_environment_warning'] = 'Inherited unrelated conflicts explicitly allowed; global pip check has NOT passed'
        stage = 'runtime_smoke'
        doctor_path = Path(args.report).with_name('metadrive_runtime_smoke.json').resolve()
        subprocess.run([sys.executable, str(_bootstrap.ROOT / 'scripts/doctor.py'), '--smoke',
                        '--source', str(source), '--output', str(doctor_path)], check=True)
        import json
        result = json.loads(doctor_path.read_text())
        report['stages'][stage] = result
        report['real_steps_executed'] = result['real_steps_executed']
        report['status'] = 'installation_and_smoke_passed_not_training_acceptance'
        return 0
    except Exception as exc:
        report.update(status='blocked_or_failed', failed_stage=stage, exception=type(exc).__name__, message=str(exc))
        return 2
    finally:
        save_json(report, args.report)
        print(f"{report.get('status', 'unknown')}; evidence: {args.report}")


if __name__ == '__main__':
    raise SystemExit(main())
