"""Create an additive venv reusing the recorded CPU base without mutating it.

This is a dependency overlay, not a fully hermetic environment. A .pth exposes
explicit base site-packages read paths; all pip writes target the new venv.
"""
from pathlib import Path
import argparse
import json
import os
import site
import subprocess
import sys
import _bootstrap
from src.utils.logger import save_json, runtime_manifest


def create(destination: Path, report_path: Path):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Use a new runtime directory; an existing runtime is never overwritten')
    records = []
    def run(command):
        p = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        records.append({'command': list(map(str, command)), 'exit_code': p.returncode,
                        'stdout': p.stdout, 'stderr': p.stderr})
        if p.returncode:
            raise RuntimeError(p.stderr[-3000:])
        return p.stdout
    report = {'status': 'started', 'base': runtime_manifest(), 'mode': 'additive_overlay_not_hermetic',
              'writes_to_base_site_packages': False, 'commands': records}
    try:
        run([sys.executable, '-m', 'venv', str(destination)])
        python = destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        child_sites = json.loads(run([str(python), '-c', 'import site,json; print(json.dumps(site.getsitepackages()))']))
        own_site = next(Path(p) for p in child_sites if Path(p).is_relative_to(destination))
        bases = [str(Path(p).resolve()) for p in site.getsitepackages() if Path(p).exists()]
        (own_site / 'recorded_cpu_base.pth').write_text('\n'.join(bases) + '\n')
        report.update(status='overlay_created_not_simulator_ready', python=str(python), base_site_paths=bases)
        run([str(python), '-c', 'import torch,numpy; print(torch.__version__,numpy.__version__)'])
    except Exception as exc:
        report.update(status='failed', exception=type(exc).__name__, message=str(exc))
        raise
    finally:
        save_json(report, report_path)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(_bootstrap.ROOT / 'external/runtime'))
    p.add_argument('--report', default=str(_bootstrap.ROOT / 'reports/metadrive_runtime_creation.json'))
    args = p.parse_args()
    print(json.dumps(create(Path(args.output), Path(args.report)), indent=2))
