"""Fresh-process MetaDrive PPO fine-tuning over a staged safety curriculum.

The command warm-starts encoder/actor weights from an existing checkpoint,
reinitializes critic/optimizer when requested via transfer, then trains each
road/traffic stage in a separate process.  Model/optimizer/counters persist
between stages; Panda3D world state does not.  This avoids treating simulator
process lifetime as part of the learned policy state.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import _bootstrap
from src.agents.transfer import transfer_actor
from src.safety.curriculum import load_safety_curriculum
from src.utils.config import load_config, save_config
from src.utils.logger import save_json, sha256, runtime_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='architecture-compatible actor warm start')
    p.add_argument('--base-config', required=True)
    p.add_argument('--curriculum-config', default=str(_bootstrap.ROOT / 'configs/safety_distillation_curriculum.yaml'))
    p.add_argument('--output', required=True)
    p.add_argument('--steps-per-stage', type=int, default=4096)
    p.add_argument('--seed', type=int, default=61)
    p.add_argument('--reference-checkpoint')
    p.add_argument('--reference-kl-coef', type=float, default=0.0)
    p.add_argument('--stage-timeout', type=float, default=240.0)
    args = p.parse_args()
    if args.steps_per_stage <= 0 or args.stage_timeout <= 0 or args.reference_kl_coef < 0:
        p.error('steps/timeouts must be positive and KL coefficient nonnegative')
    if args.reference_kl_coef > 0 and not args.reference_checkpoint:
        p.error('--reference-checkpoint is required when --reference-kl-coef > 0')

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    stages_root = output / 'stages'
    stages_root.mkdir()
    curriculum = load_safety_curriculum(args.curriculum_config)
    base = load_config(args.base_config, {'seed': args.seed, 'env': {'seed': args.seed}})
    if base.algorithm != 'ppo' or base.env.backend != 'metadrive' or base.env.observation != 'state' or base.env.num_agents != 1:
        raise ValueError('Safety curriculum fine-tuning requires single-agent state PPO on MetaDrive')
    base.reference_checkpoint = str(Path(args.reference_checkpoint).resolve()) if args.reference_checkpoint else None
    base.reference_kl_coef = float(args.reference_kl_coef)

    first = curriculum[0].apply(base.env)
    warm_config = load_config(args.base_config, {'seed': args.seed, 'env': {'seed': args.seed}})
    warm_config.env = first
    warm_config.reference_checkpoint = base.reference_checkpoint
    warm_config.reference_kl_coef = base.reference_kl_coef
    warm_config.total_steps = len(curriculum) * args.steps_per_stage
    warmstart = output / 'warmstart.pt'
    transfer = transfer_actor(args.source, warm_config, warmstart)
    manifest = {
        'status': 'started', 'source': str(Path(args.source).resolve()), 'source_sha256': sha256(args.source),
        'warmstart': str(warmstart), 'warmstart_sha256': sha256(warmstart), 'transfer': transfer,
        'curriculum': [stage.to_dict() for stage in curriculum], 'steps_per_stage': args.steps_per_stage,
        'reference_checkpoint': base.reference_checkpoint,
        'reference_kl_coef': base.reference_kl_coef, 'seed': args.seed,
        'resume_semantics': 'weights/optimizer/counter continue; MetaDrive restarts at episode boundary per stage',
        'runtime': runtime_manifest(), 'stages': [],
    }
    save_json(manifest, output / 'result.json')
    resume = warmstart
    try:
        for index, stage in enumerate(curriculum, 1):
            target_steps = index * args.steps_per_stage
            cfg = load_config(args.base_config, {'seed': args.seed, 'env': {'seed': args.seed}})
            cfg.env = stage.apply(cfg.env)
            cfg.reference_checkpoint = base.reference_checkpoint
            cfg.reference_kl_coef = base.reference_kl_coef
            cfg.total_steps = target_steps
            stage_dir = stages_root / f'{index:02d}_{stage.name}'
            stage_dir.mkdir()
            config_path = stage_dir / 'config.yaml'
            save_config(cfg, config_path)
            command = [sys.executable, str(_bootstrap.ROOT / 'scripts/train.py'), '--config', str(config_path),
                       '--output', str(stage_dir / 'run'), '--steps', str(target_steps), '--resume', str(resume)]
            log_path = stage_dir / 'process.log'
            started = time.monotonic()
            with log_path.open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=_bootstrap.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                                           timeout=args.stage_timeout)
            model = stage_dir / 'run' / 'model.pt'
            if completed.returncode != 0 or not model.is_file():
                raise RuntimeError(f'Curriculum stage {stage.name} failed; inspect {log_path}')
            row = {'index': index, 'stage': stage.to_dict(), 'target_world_steps': target_steps,
                   'elapsed_seconds': time.monotonic() - started, 'return_code': completed.returncode,
                   'model': str(model.relative_to(output)), 'model_sha256': sha256(model)}
            manifest['stages'].append(row)
            save_json(manifest, output / 'result.json')
            resume = model
        final = output / 'model.pt'
        shutil.copy2(resume, final)
        manifest.update(status='completed_not_safety_certified', final_world_steps=len(curriculum) * args.steps_per_stage,
                        final_model_sha256=sha256(final), final_model=str(final))
        save_json(manifest, output / 'result.json')
        print(json.dumps(manifest, indent=2))
    except Exception as exc:
        manifest.update(status='failed', exception=type(exc).__name__, message=str(exc))
        save_json(manifest, output / 'result.json')
        raise


if __name__ == '__main__':
    main()
