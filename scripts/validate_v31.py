"""External-CWD smoke for v3.1 learned route-adapter deployment.

Uses only previously consumed diagnostic/training-domain seeds; this validator
must never open a new release split.  It checks that the adapter can be loaded
with its bound base SHA, run in real MetaDrive, emit adapter diagnostics, and
produce a demo artifact from outside the repository working directory.
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
    adapter = root / 'models/metadrive_learned_route_adapter_v31.pt'
    # Fail before spawning external processes if the artifact/base binding is bad.
    agent = LearnedRouteAdapterAgent(model, adapter)
    records = []
    with tempfile.TemporaryDirectory(prefix='rl_autodrive_v31_') as tmp:
        cwd = Path(tmp)
        evaluation = cwd / 'evaluation'
        demo = cwd / 'adapter.gif'
        commands = [
            ('adapter_evaluate', [
                sys.executable, str(root / 'scripts/evaluate.py'), '--model', str(model),
                '--steering-adapter', str(adapter), '--episodes', '2', '--split', 'test',
                '--output', str(evaluation),
            ]),
            ('adapter_demo', [
                sys.executable, str(root / 'scripts/demo.py'), '--model', str(model),
                '--steering-adapter', str(adapter), '--seed', '31050', '--every', '8',
                '--output', str(demo),
            ]),
        ]
        for label, command in commands:
            process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=90,
                                     env={**os.environ, 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            record = {'label': label, 'command': command, 'return_code': process.returncode,
                      'stdout_tail': process.stdout[-3000:], 'stderr_tail': process.stderr[-3000:]}
            records.append(record)
            if process.returncode:
                save_json({'passed': False, 'records': records}, output)
                raise RuntimeError(f'{label} failed: {process.stderr[-1000:]}')
        report = json.loads((evaluation / 'evaluation.json').read_text())
        metrics = report['metrics']
        result = {
            'passed': True,
            'base_model_sha256': sha256(model),
            'adapter_sha256': sha256(adapter),
            'composed_fingerprint': agent.fingerprint,
            'records': records,
            'artifacts': {
                'evaluation_json': (evaluation / 'evaluation.json').is_file(),
                'failure_traces_json': (evaluation / 'failure_traces.json').is_file(),
                'demo_gif_bytes': demo.stat().st_size,
                'demo_trace_json': demo.with_suffix('.json').is_file(),
            },
            'adapter_metrics': {
                'mean_blend': metrics.get('learned_adapter_mean_blend'),
                'mean_correction': metrics.get('learned_adapter_mean_steering_correction'),
                'active_time_fraction': metrics.get('learned_adapter_active_time_fraction'),
            },
            'note': 'Uses consumed test/training-domain seeds only; not a release performance result.',
        }
        save_json(result, output)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(_bootstrap.ROOT / 'reports/v3_1/cli_smoke.json'))
    args = p.parse_args()
    print(json.dumps(run(args.output), indent=2))
