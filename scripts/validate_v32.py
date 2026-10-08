"""External-CWD deployment smoke for v3.2 without opening a new release split.

The smoke uses consumed validation11 and the fresh-process suite runner, which is
robust to MetaDrive/Panda3D interpreter-teardown stalls. It checks model/adapter
binding and a real closed-loop MetaDrive rollout from outside the repository CWD.
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
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.utils.logger import save_json, sha256


def run(output: str | Path):
    root = _bootstrap.ROOT
    model = root / 'models/metadrive_ppo_state_capability_distilled_v28.pt'
    adapter = root / 'models/metadrive_curve_route_adapter_v32.pt'
    agent = LearnedRouteAdapterAgent(model, adapter)  # validates bound base SHA
    with tempfile.TemporaryDirectory(prefix='rl_autodrive_v32_') as tmp:
        cwd = Path(tmp)
        suite = cwd / 'suite'
        command = [
            sys.executable, str(root / 'scripts/evaluate_learned_route_suite.py'),
            '--model', str(model), '--steering-adapter', str(adapter),
            '--split', 'validation11', '--suite', 'capability', '--seeds-per-stage', '1',
            '--focus-stages', 'curve_tracking', '--stage-timeout', '60', '--output', str(suite),
        ]
        proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=90,
                              env={**os.environ, 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        record = {'command':command,'return_code':proc.returncode,
                  'stdout_tail':proc.stdout[-4000:],'stderr_tail':proc.stderr[-4000:]}
        if proc.returncode:
            save_json({'passed':False,'record':record}, output)
            raise RuntimeError(f'v3.2 external-CWD suite failed: {proc.stderr[-1500:]}')
        report = json.loads((suite / 'suite_report.json').read_text())
        metrics = report['metrics']
        stage_metrics = report['stages'][0]['metrics'] if report.get('stages') else metrics
        result = {
            'passed': True,
            'base_model_sha256': sha256(model),
            'adapter_sha256': sha256(adapter),
            'composed_fingerprint': agent.fingerprint,
            'record': record,
            'evaluation': {
                'success_rate': metrics['success_rate'],
                'collision_rate': metrics['collision_rate'],
                'out_of_road_rate': metrics['out_of_road_rate'],
                'route_assist_fraction': metrics.get('route_assist_intervention_time_fraction'),
                'safety_fraction': metrics.get('safety_intervention_time_fraction'),
                'adapter_active_fraction': stage_metrics.get('learned_adapter_active_time_fraction'),
                'adapter_mean_correction': stage_metrics.get('learned_adapter_mean_steering_correction'),
            },
            'note':'Uses already-consumed validation11 only; not a new performance estimate and never opens release15/16.',
        }
        save_json(result, output)
        return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(_bootstrap.ROOT/'reports/v3_2/cli_smoke.json'))
    args=p.parse_args()
    print(json.dumps(run(args.output), indent=2))
