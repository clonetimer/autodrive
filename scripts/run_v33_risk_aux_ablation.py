"""Ablate SafetySupervisor risk auxiliary supervision on the fixed v3.3-D representation.

This protocol intentionally keeps temporal history, neighbor attention, frozen
base policy, scenario seeds and DAgger teacher fixed.  The only algorithmic
change is the risk-head loss weight.  Training uses scene_train and model
selection uses validation13; release16 is intentionally not exposed.
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


RISK_VARIANTS = {
    'R0': {'risk_loss_weight': 0.0, 'description': 'v3.3-D representation without risk auxiliary loss'},
    'R15': {'risk_loss_weight': 0.15, 'description': 'v3.3-D representation with SafetySupervisor risk auxiliary loss'},
}


def _csv_ints(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(',') if x.strip()]
    if not out or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError('training seeds must be a non-empty unique comma-separated list')
    return out


def _variants(text: str) -> list[str]:
    out = [x.strip().upper() for x in text.split(',') if x.strip()]
    bad = [x for x in out if x not in RISK_VARIANTS]
    if not out or bad or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(f'variants must be unique members of {tuple(RISK_VARIANTS)}')
    return out


def _command_plan(args) -> list[dict]:
    plan = []
    for variant in args.variants:
        v = RISK_VARIANTS[variant]
        for seed in args.training_seeds:
            name = f'{variant}_seed{seed}'
            trial = Path(args.output).resolve() / name
            train = [
                sys.executable, str(_bootstrap.ROOT / 'scripts/train_scene_aware_adapter.py'),
                '--base-model', str(Path(args.base_model).resolve()),
                '--output', str(trial / 'train'),
                '--split', 'scene_train',
                '--iterations', str(args.iterations),
                '--seeds-per-stage', str(args.seeds_per_stage),
                '--mix-betas', args.mix_betas,
                '--epochs', str(args.epochs),
                '--batch-size', str(args.batch_size),
                '--learning-rate', str(args.learning_rate),
                '--hidden-size', str(args.hidden_size),
                '--history-len', '6',
                '--neighbor-heads', str(args.neighbor_heads),
                '--max-correction', str(args.max_correction),
                '--risk-loss-weight', str(v['risk_loss_weight']),
                '--teacher-blend', str(args.teacher_blend),
                '--collection-safety', args.collection_safety,
                '--training-seed', str(seed),
                '--max-stage-steps', str(args.max_stage_steps),
                '--stage-timeout', str(args.train_stage_timeout),
                '--use-neighbors',
            ]
            adapter = trial / 'train/scene_aware_temporal_adapter.pt'
            validation = [
                sys.executable, str(_bootstrap.ROOT / 'scripts/evaluate_learned_route_suite.py'),
                '--model', str(Path(args.base_model).resolve()),
                '--split', 'validation13',
                '--suite', 'capability',
                '--seeds-per-stage', str(args.validation_seeds_per_stage),
                '--output', str(trial / 'validation13'),
                '--steering-adapter', str(adapter),
                '--stage-timeout', str(args.validation_stage_timeout),
            ]
            if args.focus_stages:
                validation += ['--focus-stages', args.focus_stages]
            plan.append({
                'name': name,
                'variant': variant,
                'training_seed': seed,
                'description': v['description'],
                'risk_loss_weight': v['risk_loss_weight'],
                'train_output': str(trial / 'train'),
                'validation_output': str(trial / 'validation13'),
                'adapter': str(adapter),
                'train_command': train,
                'validation_command': validation,
            })
    return plan


def _plan_fingerprint(manifest: dict) -> str:
    payload = {k: v for k, v in manifest.items()
               if k not in {'status', 'job_status', 'plan_fingerprint_sha256'}}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def _train_complete(job: dict) -> bool:
    return (Path(job['train_output']) / 'result.json').is_file() and Path(job['adapter']).is_file()


def _validation_complete(job: dict) -> bool:
    return (Path(job['validation_output']) / 'suite_report.json').is_file()


def _train_command_for_execution(job: dict, resume: bool) -> list[str]:
    command = list(job['train_command'])
    # A matrix-level resume may contain both partially-started and never-started
    # trials. Only the former may receive trainer --resume; passing it to a fresh
    # trial intentionally fails the trainer's run-config safety check.
    if resume and (Path(job['train_output']) / 'run_config.json').is_file():
        command.append('--resume')
    return command


def _validation_command_for_execution(job: dict, resume: bool) -> list[str]:
    command = list(job['validation_command'])
    output = Path(job['validation_output'])
    if resume and output.is_dir() and any(output.iterdir()):
        command.append('--resume')
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variants', type=_variants, default=_variants('R0,R15'))
    p.add_argument('--training-seeds', type=_csv_ints, default=_csv_ints('3300,4300,5300'))
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--seeds-per-stage', type=int, default=2)
    p.add_argument('--mix-betas', default='1.0,0.5,0.2')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--neighbor-heads', type=int, default=4)
    p.add_argument('--max-correction', type=float, default=0.18)
    p.add_argument('--teacher-blend', type=float, default=0.5)
    p.add_argument('--collection-safety', choices=('protected', 'raw'), default='protected')
    p.add_argument('--max-stage-steps', type=int, default=420)
    p.add_argument('--train-stage-timeout', type=float, default=120.0)
    p.add_argument('--validation-stage-timeout', type=float, default=120.0)
    p.add_argument('--validation-seeds-per-stage', type=int, default=3)
    p.add_argument('--focus-stages', default='')
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    base = Path(args.base_model).resolve()
    if not base.is_file():
        p.error(f'base model does not exist: {base}')
    if (args.iterations <= 0 or args.seeds_per_stage < 2 or args.validation_seeds_per_stage <= 0
            or args.max_stage_steps <= 0 or args.train_stage_timeout <= 0
            or args.validation_stage_timeout <= 0):
        p.error('iterations/validation seeds/timeouts must be positive and training seeds-per-stage must be >=2')
    if len([x for x in args.mix_betas.split(',') if x.strip()]) != args.iterations:
        p.error('mix-betas count must equal iterations')

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = _command_plan(args)
    manifest = {
        'schema': 'v33_risk_aux_ablation_v1',
        'status': 'planned' if args.dry_run else 'running',
        'base_model': str(base),
        'base_model_sha256': sha256(base),
        'training_split': 'scene_train',
        'selection_split': 'validation13',
        'release16_accessed': False,
        'fixed_representation': {'history_len': 6, 'use_neighbors': True},
        'variants': {k: RISK_VARIANTS[k] for k in args.variants},
        'training_seeds': args.training_seeds,
        'jobs': plan,
    }
    manifest['plan_fingerprint_sha256'] = _plan_fingerprint(manifest)
    manifest_path = root / 'risk_aux_ablation_manifest.json'
    prior_status = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        has_trial_data = any(Path(job['train_output']).exists() for job in plan)
        if args.resume:
            prior_fp = existing.get('plan_fingerprint_sha256')
            if prior_fp is not None and prior_fp != manifest['plan_fingerprint_sha256']:
                raise RuntimeError('Risk ablation plan fingerprint mismatch; resume into a new output directory')
            if prior_fp is None and has_trial_data:
                raise RuntimeError('Existing risk-ablation manifest predates fingerprinting; use a new output directory')
            prior_status = existing.get('job_status', {})
        elif not args.dry_run and has_trial_data:
            raise RuntimeError('Risk-ablation output already contains trial data; rerun with --resume')
    manifest['job_status'] = dict(prior_status)
    save_json(manifest, manifest_path)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    for job in plan:
        name = job['name']
        if args.resume and _train_complete(job):
            print(f"[v3.3-risk] skip completed train {name}", flush=True)
        else:
            command = _train_command_for_execution(job, args.resume)
            manifest['job_status'][name] = 'training_resume' if '--resume' in command else 'training'
            save_json(manifest, manifest_path)
            print(f"[v3.3-risk] train {name}" + (' (resume)' if '--resume' in command else ''), flush=True)
            subprocess.run(command, cwd=_bootstrap.ROOT, check=True)
        manifest['job_status'][name] = 'train_complete'
        save_json(manifest, manifest_path)

        if not args.train_only:
            if args.resume and _validation_complete(job):
                print(f"[v3.3-risk] skip completed validation13 {name}", flush=True)
            else:
                command = _validation_command_for_execution(job, args.resume)
                manifest['job_status'][name] = 'validation_resume' if '--resume' in command else 'validating'
                save_json(manifest, manifest_path)
                print(f"[v3.3-risk] validation13 {name}" + (' (resume)' if '--resume' in command else ''), flush=True)
                subprocess.run(command, cwd=_bootstrap.ROOT, check=True)
            manifest['job_status'][name] = 'validation_complete'
            save_json(manifest, manifest_path)

    manifest['status'] = 'training_complete' if args.train_only else 'validation13_complete'
    save_json(manifest, manifest_path)
    print(json.dumps({'status': manifest['status'], 'jobs': len(plan), 'release16_accessed': False}, indent=2))


if __name__ == '__main__':
    main()
