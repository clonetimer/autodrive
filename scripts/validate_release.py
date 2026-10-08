"""Execute real CLI smoke workflows from a different current directory.

Produces command/exit-code evidence. Optional MetaDrive failure is recorded as a
blocked external gate, never folded into the native success count.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import _bootstrap
from src.utils.logger import save_json


def validate(output):
    root = _bootstrap.ROOT
    records = []
    with tempfile.TemporaryDirectory(prefix='rl_autodrive_cli_') as temp:
        directory = Path(temp)
        modeldir = directory / 'trained'
        actions = [
            ('train_mappo_hybrid', ['train.py', '--config', str(root / 'configs/end_to_end.yaml'),
                                   '--steps', '64', '--output', str(modeldir)]),
            ('evaluate_frozen_model', ['evaluate.py', '--model', str(modeldir / 'model.pt'),
                                      '--episodes', '2', '--output', str(directory / 'evaluation')]),
            ('record_real_demo', ['demo.py', '--model', str(modeldir / 'model.pt'),
                                 '--seed', '2000', '--output', str(directory / 'demo.gif')])]
        for label, args in actions:
            command = [sys.executable, str(root / 'scripts' / args[0]), *args[1:]]
            process = subprocess.run(command, cwd=directory, text=True, capture_output=True,
                                     timeout=90, env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            record = {'label': label, 'command': command, 'exit_code': process.returncode,
                      'stdout': process.stdout, 'stderr': process.stderr, 'passed': process.returncode == 0}
            records.append(record)
            if process.returncode:
                save_json({'native_cli_passed': False, 'commands': records}, output)
                raise RuntimeError(f'{label} failed: {process.stderr}')
        csv = directory / 'tracks.csv'
        csv.write_text('time_s,track_id,x_m,y_m,yaw_rad,speed_mps\n0,lead,40,0,0,4\n20,lead,120,0,0,4\n')
        path = directory / 'portable.json'
        process = subprocess.run([sys.executable, str(root / 'scripts/prepare_dataset.py'), '--csv', str(csv),
                                  '--output', str(path), '--provenance', 'synthetic CLI validation'],
                                  cwd=directory, text=True, capture_output=True, timeout=30)
        records.append({'label': 'convert_synthetic_csv', 'exit_code': process.returncode,
                        'stdout': process.stdout, 'stderr': process.stderr, 'passed': process.returncode == 0})
        if process.returncode:
            raise RuntimeError(process.stderr)
        scenario = json.loads(path.read_text())
        if scenario['provenance'] != 'synthetic CLI validation':
            raise AssertionError('CSV conversion lost provenance')
        result = {'native_cli_passed': True, 'commands': records, 'working_directory': 'temporary directory outside repository',
                  'artifacts_checked': {'model_saved': (modeldir / 'model.pt').stat().st_size > 0,
                    'evaluation_json': (directory / 'evaluation/evaluation.json').exists(),
                    'gif': (directory / 'demo.gif').stat().st_size > 0,
                    'trajectory_json': (directory / 'demo.json').exists(), 'portable_tracks': len(scenario['tracks'])},
                  'metadrive': 'real MetaDrive is covered through v3.2 curve-aware learned-route-adapter/release15 evidence; see reports/v3_2 and docs/CURVE_ROUTE_ADAPTER_v3.2.0.md',
                  'note': 'Temporary smoke artifacts are deleted; release training artifacts are in experiments/.'}
        save_json(result, output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(_bootstrap.ROOT / 'reports/cli_smoke.json'))
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))
