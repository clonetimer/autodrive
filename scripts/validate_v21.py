"""Run new observation and SAC-resume CLIs outside the repository directory."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
import _bootstrap
from src.utils.logger import save_json


def validate(output):
    root = _bootstrap.ROOT
    commands = []
    with tempfile.TemporaryDirectory(prefix='autodrive_v21_cli_') as temp:
        cwd = Path(temp)
        calls = [
            ['audit_improvements.py', '--output', str(cwd / 'audit.json')],
            ['train.py', '--config', str(root / 'configs/improvement/bev_temporal.yaml'), '--steps', '64', '--output', str(cwd / 'bev')],
            ['evaluate.py', '--model', str(cwd / 'bev/model.pt'), '--split', 'validation', '--episodes', '2', '--output', str(cwd / 'eval')],
            ['train.py', '--config', str(root / 'configs/improvement/sac_resume.yaml'), '--steps', '256', '--output', str(cwd / 'sac')],
            ['train.py', '--config', str(root / 'configs/improvement/sac_resume.yaml'), '--steps', '320',
             '--resume', str(cwd / 'sac/training_state.pt'), '--output', str(cwd / 'resumed')],
        ]
        for args in calls:
            command = [sys.executable, str(root / 'scripts' / args[0]), *args[1:]]
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            commands.append({'command': command, 'exit_code': result.returncode,
                             'stdout': result.stdout, 'stderr': result.stderr})
            if result.returncode:
                save_json({'status': 'failed', 'commands': commands}, output)
                raise RuntimeError(result.stderr)
        resumed = json.loads((cwd / 'resumed/result.json').read_text())
        assert resumed['world_steps_this_run'] == 64 and resumed['world_steps_total'] == 320
        assert resumed['updates_this_run'] == 64 and resumed['parameters_changed_l1'] > 0
        result = {'status': 'passed', 'working_directory': 'fresh temporary directory outside repository',
                  'commands': commands, 'resume_result': resumed,
                  'audit_checks': json.loads((cwd / 'audit.json').read_text())['checks'],
                  'temp_artifacts_deleted': True}
        save_json(result, output)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/v2_1/new_cli_smoke.json')
    args = parser.parse_args()
    print(validate(args.output)['status'])
