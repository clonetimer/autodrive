"""DAgger-style MetaDrive SafetySupervisor correction distillation.

v2.6 can collect on a staged real-MetaDrive curriculum.  Each round executes
*the current actor* through the supervisor, appends newly labelled states to the
aggregate dataset, and then distils again.  Frozen release splits are never used
for collection or hyper-parameter selection.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import _bootstrap
from src.agents.trainer import load_checkpoint, save_checkpoint
from src.envs.metadrive_wrapper import MetaDriveParallelEnv
from src.safety.curriculum import load_safety_curriculum
from src.safety.distillation import distill_safety_actor
from src.safety.lane_teacher import lane_teacher_action
from src.scenarios.generator import SPLITS
from src.utils.config import load_config
from src.utils.logger import save_json, sha256, runtime_manifest


def seed_block(start: int, count: int) -> list[int]:
    lo, hi = SPLITS['safety_train']
    begin = lo + start
    if count <= 0 or begin < lo or begin + count - 1 > hi:
        raise ValueError('Requested safety-training seed block is out of range')
    return list(range(begin, begin + count))


def collect(policy, env_config, seeds: list[int], max_steps_per_seed: int, stage_name: str = 'base', *, lane_stabilizer: bool = False, lane_blend_max: float = 0.70):
    env = MetaDriveParallelEnv(env_config)
    rows = []
    try:
        for seed in seeds:
            observations, _ = env.reset(seed=seed)
            steps = 0
            while env.agents and steps < max_steps_per_seed:
                state = np.asarray(observations['agent0'], dtype=np.float32)
                proposed = policy.predict(state, deterministic=True)
                next_obs, rewards, terms, truncs, infos = env.step({'agent0': proposed})
                info = infos['agent0']
                applied = np.asarray(info['applied_action'], dtype=np.float32)
                target = applied.copy()
                lane = lane_teacher_action(proposed, state, max_blend=lane_blend_max) if lane_stabilizer else None
                if lane is not None:
                    target[0] = lane.target_steer
                safety_intervened = bool(info['safety_intervention'])
                lane_intervened = bool(lane.intervened) if lane is not None else False
                safety_severity = float(info['safety_severity'])
                lane_severity = float(lane.severity) if lane is not None else 0.0
                rows.append({
                    'state': state.copy(), 'target': target.copy(), 'proposed': proposed.copy(),
                    'intervened': bool(safety_intervened or lane_intervened),
                    'severity': max(safety_severity, lane_severity),
                    'safety_intervened': safety_intervened, 'lane_intervened': lane_intervened,
                    'safety_severity': safety_severity, 'lane_severity': lane_severity,
                    'correction': float(abs(applied[1] - proposed[1])),
                    'lateral_correction': float(abs(target[0] - proposed[0])), 'seed': int(seed),
                    'step': int(info['step']), 'collision': bool(info['collision']),
                    'success': bool(info['success']), 'route_completion': float(info['route_completion']),
                    'ttc_s': info.get('risk', {}).get('ttc_s'), 'stage': stage_name,
                })
                steps += 1
                if terms['agent0'] or truncs['agent0']:
                    break
                observations = {'agent0': next_obs['agent0']}
        if not rows:
            raise RuntimeError('No supervisor-distillation transitions were collected')
        return rows
    finally:
        env.close()


def arrays(rows):
    return dict(
        state=np.stack([r['state'] for r in rows]).astype(np.float32),
        target_action=np.stack([r['target'] for r in rows]).astype(np.float32),
        proposed_action=np.stack([r['proposed'] for r in rows]).astype(np.float32),
        intervened=np.asarray([r['intervened'] for r in rows], dtype=bool),
        severity=np.asarray([r['severity'] for r in rows], dtype=np.float32),
        correction=np.asarray([r['correction'] for r in rows], dtype=np.float32),
        lateral_correction=np.asarray([r.get('lateral_correction', 0.0) for r in rows], dtype=np.float32),
        safety_intervened=np.asarray([r.get('safety_intervened', r['intervened']) for r in rows], dtype=bool),
        lane_intervened=np.asarray([r.get('lane_intervened', False) for r in rows], dtype=bool),
        seed=np.asarray([r['seed'] for r in rows], dtype=np.int32),
        step=np.asarray([r['step'] for r in rows], dtype=np.int32),
        stage=np.asarray([r['stage'] for r in rows], dtype='U32'),
    )


def collection_summary(rows):
    correction = np.asarray([r['correction'] for r in rows], dtype=np.float64)
    lateral = np.asarray([r.get('lateral_correction', 0.0) for r in rows], dtype=np.float64)
    intervened = np.asarray([r['intervened'] for r in rows], dtype=bool)
    safety = np.asarray([r.get('safety_intervened', r['intervened']) for r in rows], dtype=bool)
    lane = np.asarray([r.get('lane_intervened', False) for r in rows], dtype=bool)
    return {
        'samples': len(rows), 'unique_seeds': len({r['seed'] for r in rows}),
        'stages': sorted({r['stage'] for r in rows}),
        'intervention_samples': int(intervened.sum()),
        'intervention_fraction': float(intervened.mean()),
        'safety_intervention_samples': int(safety.sum()),
        'safety_intervention_fraction': float(safety.mean()),
        'lane_intervention_samples': int(lane.sum()),
        'lane_intervention_fraction': float(lane.mean()),
        'mean_longitudinal_correction': float(correction.mean()),
        'p95_longitudinal_correction': float(np.quantile(correction, .95)),
        'mean_lateral_correction': float(lateral.mean()),
        'p95_lateral_correction': float(np.quantile(lateral, .95)),
        'collisions_observed_under_supervision': int(sum(r['collision'] for r in rows)),
        'success_terminal_steps': int(sum(r['success'] for r in rows)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='single-agent state PPO checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--collection-config', help='optional TrainConfig whose env defines base MetaDrive collection settings')
    p.add_argument('--curriculum-config', help='optional staged safety-collection YAML')
    p.add_argument('--iterations', type=int, default=2)
    p.add_argument('--seed-offset', type=int, default=0, help='offset inside the safety_train split')
    p.add_argument('--seeds-per-iteration', type=int, default=12)
    p.add_argument('--max-steps-per-seed', type=int, default=180)
    p.add_argument('--epochs-per-iteration', type=int, default=18)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--intervention-weight', type=float, default=4.0)
    p.add_argument('--steering-weight', type=float, default=0.35)
    p.add_argument('--freeze-encoder', action='store_true',
                   help='do not change shared observation features during safety imitation')
    p.add_argument('--longitudinal-only', action='store_true',
                   help='update only actor longitudinal row; requires --freeze-encoder')
    p.add_argument('--ttc-threshold', type=float, default=1.8)
    p.add_argument('--risk-threshold', type=float, default=0.68)
    p.add_argument('--lane-stabilizer', action='store_true', help='add a training-only policy-visible pure-pursuit steering teacher')
    p.add_argument('--lane-blend-max', type=float, default=0.70)
    args = p.parse_args()
    if min(args.iterations, args.seeds_per_iteration, args.max_steps_per_seed, args.epochs_per_iteration,
           args.batch_size) <= 0 or args.seed_offset < 0:
        p.error('count arguments must be positive and seed-offset nonnegative')
    if args.longitudinal_only and not args.freeze_encoder:
        p.error('--longitudinal-only requires --freeze-encoder')
    if args.lane_stabilizer and args.longitudinal_only:
        p.error('--lane-stabilizer cannot be combined with --longitudinal-only because steering labels would be frozen')
    if not 0 < args.lane_blend_max <= 1:
        p.error('--lane-blend-max must lie in (0,1]')

    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    model, config, payload = load_checkpoint(args.source)
    if config.env.backend != 'metadrive' or config.env.observation != 'state' or config.env.num_agents != 1:
        raise ValueError('Safety distillation requires a single-agent MetaDrive state policy')
    base_collection = copy.deepcopy(load_config(args.collection_config).env if args.collection_config else config.env)
    if base_collection.backend != 'metadrive' or base_collection.observation != 'state' or base_collection.num_agents != 1:
        raise ValueError('Collection config must be single-agent MetaDrive state')
    deploy_config = copy.deepcopy(config)
    deploy_config.env = copy.deepcopy(base_collection)
    deploy_config.env.safety_supervisor = False
    base_collection.safety_supervisor = True
    base_collection.safety_ttc_threshold_s = args.ttc_threshold
    base_collection.safety_risk_threshold = args.risk_threshold
    stages = load_safety_curriculum(args.curriculum_config) if args.curriculum_config else None
    if stages and args.iterations > len(stages):
        raise ValueError('iterations cannot exceed the number of curriculum stages')

    save_json(runtime_manifest(), out / 'runtime.json')
    all_rows, iterations = [], []
    for iteration in range(args.iterations):
        stage = stages[iteration] if stages else None
        collection_config = stage.apply(base_collection) if stage else copy.deepcopy(base_collection)
        collection_config.safety_supervisor = True
        collection_config.safety_ttc_threshold_s = args.ttc_threshold
        collection_config.safety_risk_threshold = args.risk_threshold
        offset = args.seed_offset + iteration * args.seeds_per_iteration
        seeds = seed_block(offset, args.seeds_per_iteration)
        stage_name = stage.name if stage else 'base'
        rows = collect(model, collection_config, seeds, args.max_steps_per_seed, stage_name,
                       lane_stabilizer=args.lane_stabilizer, lane_blend_max=args.lane_blend_max)
        for row in rows:
            row['iteration'] = iteration
        all_rows.extend(rows)
        data = arrays(all_rows)
        result = distill_safety_actor(
            model, data['state'], data['target_action'], data['intervened'], data['severity'],
            epochs=args.epochs_per_iteration, batch_size=args.batch_size,
            learning_rate=args.learning_rate, intervention_weight=args.intervention_weight,
            steering_weight=args.steering_weight, train_encoder=not args.freeze_encoder,
            longitudinal_only=args.longitudinal_only, device=config.device, seed=config.seed + iteration)
        iteration_evidence = {
            'iteration': iteration, 'stage': stage.to_dict() if stage else None,
            'collection_env': collection_config.__dict__, 'collection_seeds': seeds,
            'collection': collection_summary(rows), 'aggregate_samples': len(all_rows),
            'distillation': result.to_dict(),
        }
        iterations.append(iteration_evidence)
        save_checkpoint(out / f'iteration_{iteration + 1}.pt', model, deploy_config, optimizer=None,
                        steps=int(payload.get('steps', 0)), extra={'safety_distillation': iteration_evidence})

    data = arrays(all_rows)
    np.savez_compressed(out / 'safety_distillation_dataset.npz', **data,
                        iteration=np.asarray([r['iteration'] for r in all_rows], dtype=np.int16))
    provenance = {
        'method': 'dagger_style_supervisor_distillation_with_optional_lane_teacher',
        'source_model': str(Path(args.source)), 'source_sha256': sha256(args.source),
        'source_rl_world_steps': int(payload.get('steps', 0)),
        'base_collection_env': base_collection.__dict__,
        'curriculum': [stage.to_dict() for stage in stages] if stages else None,
        'safety_train_seed_offset': args.seed_offset,
        'supervisor_changes_steering': False,
        'lane_teacher_enabled': bool(args.lane_stabilizer), 'lane_teacher_max_blend': float(args.lane_blend_max),
        'supervisor_thresholds': {'ttc_s': args.ttc_threshold, 'risk': args.risk_threshold},
        'distillation_scope': {'train_encoder': not args.freeze_encoder,
                               'longitudinal_only': args.longitudinal_only},
        'iterations': iterations, 'aggregate_collection': collection_summary(all_rows),
        'critic_trained': False, 'log_std_trained': False,
        'deployment_safety_supervisor_enabled': False,
        'claim_boundary': 'imitation of heuristic longitudinal corrections and optional training-only lane teacher; not a safety certificate',
    }
    final_path = out / 'model.pt'
    save_checkpoint(final_path, model, deploy_config, optimizer=None, steps=int(payload.get('steps', 0)),
                    extra={'safety_distillation': provenance})
    provenance['model_sha256'] = sha256(final_path)
    save_json(provenance, out / 'result.json')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
