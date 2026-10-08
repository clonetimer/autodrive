"""Evaluate a frozen state policy plus optional learned route adapter on a MetaDrive suite.

Each stage runs in a fresh child process. MetaDrive/Panda3D occasionally leaves
long-lived process state after repeated environment construction, so the parent
accepts a stage as soon as its report is atomically present and then reaps the
worker. This keeps release evaluation bounded and reproducible.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
import _bootstrap
from src.agents.trainer import load_checkpoint
from src.end_to_end.agent import EndToEndAgent
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.navigation.longitudinal_residual_adapter import FactorizedDrivingAgent
from src.navigation.gated_longitudinal_adapter import GatedFactorizedDrivingAgent
from src.evaluation.learned_adapter_evaluator import evaluate
from src.evaluation.learned_adapter_metrics import summarize
from src.evaluation.safety_suite import SafetySuiteSpec, standard_safety_suite
from src.evaluation.path_suite import standard_path_suite
from src.evaluation.capability_suite import standard_capability_suite
from src.evaluation.interaction_suite import interaction_stress_suite
from src.utils.logger import save_json, sha256


def _factorized_agent(model: str, lateral_adapter: str, longitudinal_adapter: str):
    """Load either the historical v4.1 or authority-gated v4.2 artifact."""
    payload = torch.load(longitudinal_adapter, map_location='cpu', weights_only=False)
    fmt = payload.get('format') if isinstance(payload, dict) else None
    if fmt == 'gated_longitudinal_temporal_adapter_v42':
        return GatedFactorizedDrivingAgent(model, lateral_adapter, longitudinal_adapter)
    if fmt == 'longitudinal_temporal_adapter_v41':
        return FactorizedDrivingAgent(model, lateral_adapter, longitudinal_adapter)
    raise ValueError(f'Unsupported factorized longitudinal artifact format: {fmt}')

def _terminate(process: subprocess.Popen, grace: float = 2.0) -> str:
    if process.poll() is not None:
        return 'already_exited'
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return 'already_exited'
    try:
        process.wait(timeout=grace)
        return 'terminated_group'
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return 'killed_group'


def _env_for_model(model: str, supervised: bool, ttc: float, risk: float, route_assist: bool=False, route_blend: float=1.0, route_mode: str='fixed', route_min_blend: float=0.0):
    _, config, _ = load_checkpoint(model)
    if config.env.backend != 'metadrive' or config.env.observation != 'state' or config.env.num_agents != 1:
        raise ValueError('Safety suite requires a single-agent MetaDrive state checkpoint')
    env = config.env
    env.safety_supervisor = supervised
    env.safety_ttc_threshold_s = ttc
    env.safety_risk_threshold = risk
    env.route_assist = bool(route_assist)
    env.route_assist_blend = float(route_blend)
    env.route_assist_mode = str(route_mode)
    env.route_assist_min_blend = float(route_min_blend)
    return env


def _worker(args) -> int:
    payload = json.loads(args.worker_spec_json)
    payload['seeds'] = tuple(payload['seeds'])
    spec = SafetySuiteSpec(**payload)
    env = spec.env(_env_for_model(args.model, args.supervised, args.ttc_threshold, args.risk_threshold, args.route_assist, args.route_blend, args.route_mode, args.route_min_blend))
    if args.longitudinal_adapter:
        agent = _factorized_agent(args.model, args.lateral_adapter, args.longitudinal_adapter)
    else:
        agent = (LearnedRouteAdapterAgent(args.model, args.steering_adapter)
                 if args.steering_adapter else EndToEndAgent(args.model))
    report = evaluate(lambda name, seed: agent, env, list(spec.seeds), output=args.output,
                      label=Path(args.model).stem + ('+supervisor' if args.supervised else '') + ':' + spec.name)
    print(json.dumps(report['metrics'], indent=2))
    return 0


def _merge(root: Path, suite, label: str, stage_runtime: list[dict]) -> dict:
    episodes, stages, total_steps, elapsed = [], [], 0, 0.0
    for spec in suite:
        report = json.loads((root / spec.name / 'evaluation.json').read_text(encoding='utf-8'))
        for row in report['episodes']:
            episodes.append({**row, 'suite_stage': spec.name, 'metadrive_map': spec.metadrive_map,
                             'traffic_density': spec.traffic_density})
        stages.append({'spec': spec.to_dict(), 'metrics': report['metrics'],
                       'world_steps': report['world_steps'], 'elapsed_seconds': report['elapsed_seconds']})
        total_steps += int(report['world_steps'])
        elapsed += float(report['elapsed_seconds'])
    result = {'label': label, 'suite': [spec.to_dict() for spec in suite], 'metrics': summarize(episodes),
              'stages': stages, 'episodes': episodes, 'world_steps': total_steps, 'elapsed_seconds': elapsed,
              'closed_loop_world_steps_per_second': total_steps / max(elapsed, 1e-9),
              'worker_runtime': stage_runtime}
    save_json(result, root / 'suite_report.json')
    return result




def _stage_request(model: str, spec: SafetySuiteSpec, supervised: bool, ttc: float, risk: float, route_assist: bool=False, route_blend: float=1.0, route_mode: str='fixed', route_min_blend: float=0.0, steering_adapter: str | None=None, lateral_adapter: str | None=None, longitudinal_adapter: str | None=None) -> dict:
    return {
        'model': str(Path(model).resolve()),
        'model_sha256': sha256(model),
        'supervised': bool(supervised),
        'ttc_threshold': float(ttc),
        'risk_threshold': float(risk),
        'route_assist': bool(route_assist),
        'route_blend': float(route_blend),
        'route_mode': str(route_mode),
        'route_min_blend': float(route_min_blend),
        'steering_adapter': str(Path(steering_adapter).resolve()) if steering_adapter else None,
        'steering_adapter_sha256': sha256(steering_adapter) if steering_adapter else None,
        'lateral_adapter': str(Path(lateral_adapter).resolve()) if lateral_adapter else None,
        'lateral_adapter_sha256': sha256(lateral_adapter) if lateral_adapter else None,
        'longitudinal_adapter': str(Path(longitudinal_adapter).resolve()) if longitudinal_adapter else None,
        'longitudinal_adapter_sha256': sha256(longitudinal_adapter) if longitudinal_adapter else None,
        'spec': spec.to_dict(),
    }


def _stage_committed(stage_dir: Path, expected: dict) -> bool:
    try:
        request = json.loads((stage_dir / 'stage_request.json').read_text(encoding='utf-8'))
        marker = json.loads((stage_dir / 'evaluation.complete.json').read_text(encoding='utf-8'))
        report = json.loads((stage_dir / 'evaluation.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    if request != expected or marker.get('status') != 'complete':
        return False
    spec = expected['spec']
    env = report.get('env_config', {})
    return (report.get('seeds') == spec['seeds'] and
            env.get('metadrive_map') == spec['metadrive_map'] and
            float(env.get('metadrive_traffic_density', -1)) == float(spec['traffic_density']) and
            float(env.get('target_speed', -1)) == float(spec['target_speed']) and
            int(env.get('max_steps', -1)) == int(spec['max_steps']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--split', choices=['validation', 'validation2', 'validation3', 'validation4', 'validation5', 'validation6', 'release5', 'release6', 'release7', 'release8', 'release9', 'release10', 'validation7', 'release11', 'validation8', 'release12', 'validation9', 'release13', 'validation10', 'release14', 'validation11', 'release15', 'validation12', 'release16', 'validation13', 'joint_train', 'validation14', 'longitudinal_train', 'validation15', 'authority_train', 'validation16'], default='validation')
    p.add_argument('--suite', choices=['safety', 'path', 'capability', 'interaction'], default='safety')
    p.add_argument('--seeds-per-stage', type=int, default=10)
    p.add_argument('--focus-stages', default='',
                   help='comma-separated suite stages to evaluate; empty evaluates the full suite')
    p.add_argument('--output', required=True)
    p.add_argument('--supervised', action='store_true')
    p.add_argument('--route-assist', action='store_true', help='transparent policy-visible pure-pursuit steering assist')
    p.add_argument('--route-blend', type=float, default=1.0)
    p.add_argument('--route-mode', choices=['fixed', 'adaptive'], default='fixed')
    p.add_argument('--route-min-blend', type=float, default=0.0)
    p.add_argument('--steering-adapter', help='learned residual steering/joint adapter; no geometric RouteAssist call at inference')
    p.add_argument('--lateral-adapter', help='frozen lateral adapter for v4.1 factorized deployment')
    p.add_argument('--longitudinal-adapter', help='v4.1 longitudinal-only adapter; requires --lateral-adapter')
    p.add_argument('--ttc-threshold', type=float, default=1.8)
    p.add_argument('--risk-threshold', type=float, default=.68)
    p.add_argument('--stage-timeout', type=float, default=120.0)
    p.add_argument('--resume', action='store_true', help='reuse only fingerprint-matched committed stage reports')
    p.add_argument('--worker-spec-json', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_spec_json:
        # MetaDrive/Panda3D can keep non-daemon engine state alive after a worker
        # has already closed the environment and atomically written its report.
        # This worker owns a dedicated process, so terminate it immediately after
        # successful evaluation instead of letting interpreter teardown stall a
        # release suite for minutes.  All persistent evidence is flushed first.
        code = _worker(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    if (args.seeds_per_stage <= 0 or args.stage_timeout <= 0
            or not 0 <= args.route_min_blend <= args.route_blend <= 1):
        p.error('suite/timeout must be positive and route blends satisfy 0 <= min <= max <= 1')

    # Validate the model once before spawning workers.
    _env_for_model(args.model, args.supervised, args.ttc_threshold, args.risk_threshold, args.route_assist, args.route_blend, args.route_mode, args.route_min_blend)
    if bool(args.lateral_adapter) != bool(args.longitudinal_adapter):
        p.error('--lateral-adapter and --longitudinal-adapter must be supplied together')
    if args.steering_adapter and args.longitudinal_adapter:
        p.error('Use either --steering-adapter or the v4.1 factorized adapter pair, not both')
    if args.longitudinal_adapter:
        _factorized_agent(args.model, args.lateral_adapter, args.longitudinal_adapter)
    elif args.steering_adapter:
        # Load once in parent so a mismatched base SHA fails before workers spawn.
        LearnedRouteAdapterAgent(args.model, args.steering_adapter)
    if args.suite == 'safety':
        suite = standard_safety_suite(args.split, args.seeds_per_stage)
    elif args.suite == 'path':
        suite = standard_path_suite(args.split, args.seeds_per_stage)
    elif args.suite == 'interaction':
        suite = interaction_stress_suite(args.split, args.seeds_per_stage)
    else:
        suite = standard_capability_suite(args.split, args.seeds_per_stage)
    if args.focus_stages:
        requested = {item.strip() for item in args.focus_stages.split(',') if item.strip()}
        known = {spec.name for spec in suite}
        unknown = requested - known
        if unknown:
            p.error(f'Unknown focus stages: {sorted(unknown)}')
        suite = tuple(spec for spec in suite if spec.name in requested)
        if not suite:
            p.error('focus-stages selected no suite stages')
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError('Choose an empty safety-suite output directory or use --resume')
    root.mkdir(parents=True, exist_ok=True)
    runtime = []
    for spec in suite:
        stage_dir = root / spec.name
        expected_request = _stage_request(args.model, spec, args.supervised, args.ttc_threshold, args.risk_threshold, args.route_assist, args.route_blend, args.route_mode, args.route_min_blend, args.steering_adapter, args.lateral_adapter, args.longitudinal_adapter)
        if args.resume and stage_dir.is_dir() and _stage_committed(stage_dir, expected_request):
            runtime.append({'stage': spec.name, 'elapsed_seconds': 0.0, 'return_code': 0, 'timed_out': False,
                            'termination': 'reused_committed_report', 'report_complete': True})
            continue
        if stage_dir.exists():
            if not args.resume:
                raise FileExistsError(f'Stage output already exists: {stage_dir}')
            import shutil
            shutil.rmtree(stage_dir)
        stage_dir.mkdir()
        save_json(expected_request, stage_dir / 'stage_request.json')
        log_path = stage_dir / 'worker.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--model', str(Path(args.model).resolve()),
                   '--output', str(stage_dir), '--worker-spec-json', json.dumps(spec.to_dict()),
                   '--ttc-threshold', str(args.ttc_threshold), '--risk-threshold', str(args.risk_threshold),
                   '--suite', args.suite, '--route-blend', str(args.route_blend),
                   '--route-mode', args.route_mode, '--route-min-blend', str(args.route_min_blend)]
        if args.supervised:
            command.append('--supervised')
        if args.route_assist:
            command.append('--route-assist')
        if args.steering_adapter:
            command += ['--steering-adapter', str(Path(args.steering_adapter).resolve())]
        if args.longitudinal_adapter:
            command += ['--lateral-adapter', str(Path(args.lateral_adapter).resolve()),
                        '--longitudinal-adapter', str(Path(args.longitudinal_adapter).resolve())]
        started = time.monotonic()
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=_bootstrap.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                                       start_new_session=True)
            termination = 'not_needed'
            timed_out = False
            report_path = stage_dir / 'evaluation.json'
            commit_path = stage_dir / 'evaluation.complete.json'
            while True:
                # The evaluation report is the stage commit point. MetaDrive's
                # env.close()/Panda3D teardown may hang after that commit, so do
                # not wait for natural interpreter shutdown. Validate the JSON,
                # terminate the isolated worker group immediately, then merge.
                if commit_path.is_file():
                    try:
                        marker = json.loads(commit_path.read_text(encoding='utf-8'))
                        if marker.get('status') != 'complete':
                            raise ValueError('incomplete marker')
                        json.loads(report_path.read_text(encoding='utf-8'))
                    except (json.JSONDecodeError, OSError, ValueError):
                        pass
                    else:
                        termination = _terminate(process, grace=.5)
                        break
                if process.poll() is not None:
                    break
                if time.monotonic() - started > args.stage_timeout:
                    timed_out = True
                    termination = _terminate(process, grace=.5)
                    break
                time.sleep(.1)
        ok = (stage_dir / 'evaluation.json').is_file() and (stage_dir / 'evaluation.complete.json').is_file()
        runtime.append({'stage': spec.name, 'elapsed_seconds': time.monotonic() - started,
                        'return_code': process.poll(), 'timed_out': timed_out,
                        'termination': termination, 'report_complete': ok})
        if not ok:
            save_json({'status': 'failed', 'worker_runtime': runtime}, root / 'failure.json')
            raise RuntimeError(f'Safety-suite stage {spec.name} failed; inspect {log_path}')
    label = Path(args.model).stem + ('+factorized-v41' if args.longitudinal_adapter else '+adapter' if args.steering_adapter else '') + ('+route' if args.route_assist else '') + ('+supervisor' if args.supervised else '')
    result = _merge(root, suite, label, runtime)
    print(json.dumps({'metrics': result['metrics'], 'stages': result['stages'], 'worker_runtime': runtime}, indent=2))


if __name__ == '__main__':
    main()
