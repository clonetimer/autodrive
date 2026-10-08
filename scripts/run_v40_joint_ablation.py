"""Train the frozen v4.0 P0/P1/M0/M1 factorial matrix on joint_train only.

This launcher intentionally has NO validation14 or release17 execution path.
Training must be completed and frozen before a separate holdout-opening script
is used.  This prevents an interrupted training matrix from gradually peeking at
selection results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import _bootstrap
from src.utils.logger import save_json, sha256


VARIANTS = {
    'P0': {'collection_mode': 'protected', 'enable_longitudinal': False,
           'description': 'protected DAgger, steering-only residual'},
    'P1': {'collection_mode': 'protected', 'enable_longitudinal': True,
           'description': 'protected DAgger, joint steering+longitudinal residual'},
    'M0': {'collection_mode': 'mixed', 'enable_longitudinal': False,
           'description': 'protected warm-start then raw learner support, steering-only residual'},
    'M1': {'collection_mode': 'mixed', 'enable_longitudinal': True,
           'description': 'protected warm-start then raw learner support, joint residual'},
}


def _csv_ints(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(',') if x.strip()]
    if not out or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError('training seeds must be non-empty and unique')
    return out


def _variants(text: str) -> list[str]:
    out = [x.strip().upper() for x in text.split(',') if x.strip()]
    if not out or len(set(out)) != len(out) or any(x not in VARIANTS for x in out):
        raise argparse.ArgumentTypeError(f'variants must be unique members of {tuple(VARIANTS)}')
    return out


def _command_plan(args) -> list[dict]:
    plan = []
    for variant in args.variants:
        v = VARIANTS[variant]
        for seed in args.training_seeds:
            name = f'{variant}_seed{seed}'
            trial = Path(args.output).resolve() / name
            cmd = [
                sys.executable, str(_bootstrap.ROOT / 'scripts/train_joint_action_adapter.py'),
                '--base-model', str(Path(args.base_model).resolve()),
                '--output', str(trial / 'train'),
                '--split', 'joint_train',
                '--iterations', str(args.iterations),
                '--seeds-per-stage', str(args.seeds_per_stage),
                '--mix-betas', args.mix_betas,
                '--epochs', str(args.epochs),
                '--batch-size', str(args.batch_size),
                '--learning-rate', str(args.learning_rate),
                '--hidden-size', str(args.hidden_size),
                '--history-len', str(args.history_len),
                '--neighbor-heads', str(args.neighbor_heads),
                '--use-neighbors',
                '--max-steering-correction', str(args.max_steering_correction),
                '--max-longitudinal-correction', str(args.max_longitudinal_correction),
                '--longitudinal-loss-weight', str(args.longitudinal_loss_weight),
                '--teacher-blend', str(args.teacher_blend),
                '--collection-mode', v['collection_mode'],
                '--training-seed', str(seed),
                '--max-stage-steps', str(args.max_stage_steps),
                '--stage-timeout', str(args.stage_timeout),
                '--torch-threads', str(args.torch_threads),
                '--enable-longitudinal' if v['enable_longitudinal'] else '--no-enable-longitudinal',
            ]
            plan.append({
                'name': name,
                'variant': variant,
                'training_seed': seed,
                'description': v['description'],
                'collection_mode': v['collection_mode'],
                'enable_longitudinal': v['enable_longitudinal'],
                'train_output': str(trial / 'train'),
                'adapter': str(trial / 'train/joint_action_temporal_adapter.pt'),
                'train_command': cmd,
            })
    return plan


def _plan_fingerprint(manifest: dict) -> str:
    payload = {k: v for k, v in manifest.items()
               if k not in {'status', 'job_status', 'plan_fingerprint_sha256'}}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def _complete(job: dict) -> bool:
    return (Path(job['train_output']) / 'result.json').is_file() and Path(job['adapter']).is_file()


def _execution_command(job: dict, resume: bool) -> list[str]:
    cmd = list(job['train_command'])
    if resume and (Path(job['train_output']) / 'run_config.json').is_file():
        cmd.append('--resume')
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variants', type=_variants, default=_variants('P0,P1,M0,M1'))
    p.add_argument('--training-seeds', type=_csv_ints, default=_csv_ints('6400,7400,8400'))
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--seeds-per-stage', type=int, default=1)
    p.add_argument('--mix-betas', default='1.0,0.5,0.2')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--history-len', type=int, default=6)
    p.add_argument('--neighbor-heads', type=int, default=4)
    p.add_argument('--max-steering-correction', type=float, default=.18)
    p.add_argument('--max-longitudinal-correction', type=float, default=1.5)
    p.add_argument('--longitudinal-loss-weight', type=float, default=1.0)
    p.add_argument('--teacher-blend', type=float, default=.5)
    p.add_argument('--max-stage-steps', type=int, default=420)
    p.add_argument('--stage-timeout', type=float, default=150.0)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    base = Path(args.base_model).resolve()
    if not base.is_file():
        p.error(f'base model does not exist: {base}')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = _command_plan(args)
    manifest = {
        'schema': 'v40_joint_factorial_training_v1',
        'status': 'planned' if args.dry_run else 'running',
        'base_model': str(base),
        'base_model_sha256': sha256(base),
        'training_split': 'joint_train',
        'selection_split_reserved': 'validation14',
        'final_release_reserved': 'release17',
        'validation14_accessed': False,
        'release17_accessed': False,
        'factorial_axes': {
            'collection_support': ['protected', 'mixed'],
            'longitudinal_supervision': [False, True],
        },
        'variants': {k: VARIANTS[k] for k in args.variants},
        'training_seeds': args.training_seeds,
        'jobs': plan,
    }
    manifest['plan_fingerprint_sha256'] = _plan_fingerprint(manifest)
    path = root / 'v40_training_manifest.json'
    prior_status = {}
    if path.is_file():
        existing = json.loads(path.read_text())
        has_data = any(Path(job['train_output']).exists() for job in plan)
        if args.resume:
            if existing.get('plan_fingerprint_sha256') != manifest['plan_fingerprint_sha256']:
                raise RuntimeError('v4 training plan mismatch; resume into a new output directory')
            prior_status = existing.get('job_status', {})
        elif not args.dry_run and has_data:
            raise RuntimeError('Training output already contains trial data; use --resume')
    manifest['job_status'] = dict(prior_status)
    save_json(manifest, path)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    for job in plan:
        name = job['name']
        if args.resume and _complete(job):
            print(f'[v4.0] skip completed train {name}', flush=True)
        else:
            cmd = _execution_command(job, args.resume)
            manifest['job_status'][name] = 'training_resume' if '--resume' in cmd else 'training'
            save_json(manifest, path)
            print(f'[v4.0] train {name}' + (' (resume)' if '--resume' in cmd else ''), flush=True)
            subprocess.run(cmd, cwd=_bootstrap.ROOT, check=True)
        manifest['job_status'][name] = 'train_complete'
        save_json(manifest, path)

    manifest['status'] = 'training_complete'
    save_json(manifest, path)
    print(json.dumps({'status': 'training_complete', 'jobs': len(plan),
                      'validation14_accessed': False, 'release17_accessed': False}, indent=2))


if __name__ == '__main__':
    main()
