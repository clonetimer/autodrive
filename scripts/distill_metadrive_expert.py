"""Distil a privileged MetaDrive IDM steering expert into a learned state policy.

The expert can read simulator-internal navigation/lidar state and is therefore
*training-only*.  The student still receives exactly the normal project state
observation at deployment.  This script is deliberately separated from raw
policy evaluation so privileged teacher access cannot leak into release claims.
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
from src.experts.metadrive_idm import MetaDriveIDMTeacher
from src.safety.curriculum import load_safety_curriculum
from src.safety.distillation import distill_safety_actor
from src.scenarios.generator import SPLITS
from src.utils.config import load_config
from src.utils.logger import runtime_manifest, save_json, sha256


def expert_seed_block(start: int, count: int) -> list[int]:
    lo, hi = SPLITS['expert_train']
    begin = lo + start
    if count <= 0 or begin < lo or begin + count - 1 > hi:
        raise ValueError('Requested expert-training seed block is out of range')
    return list(range(begin, begin + count))


def collect(policy, env_config, seeds: list[int], max_steps_per_seed: int, stage_name: str,
            *, expert_blend: float = 1.0) -> list[dict]:
    if not 0 < expert_blend <= 1:
        raise ValueError('expert_blend must lie in (0,1]')
    env = MetaDriveParallelEnv(env_config)
    rows = []
    try:
        for seed in seeds:
            observations, _ = env.reset(seed=seed)
            teacher = MetaDriveIDMTeacher(env, 'agent0', seed=seed, target_speed_mps=env_config.target_speed)
            steps = 0
            while env.agents and steps < max_steps_per_seed:
                state = np.asarray(observations['agent0'], dtype=np.float32)
                proposed = policy.predict(state, deterministic=True)
                expert = teacher.decide()
                executed = proposed.copy()
                executed[0] = float(np.clip((1 - expert_blend) * proposed[0] + expert_blend * expert.steering, -1, 1))
                next_obs, _, terms, truncs, infos = env.step({'agent0': executed})
                info = infos['agent0']
                applied = np.asarray(info['applied_action'], dtype=np.float32)
                lateral_delta = float(abs(expert.steering - proposed[0]))
                longitudinal_delta = float(abs(applied[1] - proposed[1]))
                intervened = bool(lateral_delta > .02 or info.get('safety_intervention', False))
                severity = float(np.clip(max(lateral_delta, float(info.get('safety_severity', 0.0))), 0, 1))
                rows.append({
                    'state': state.copy(), 'proposed_action': proposed.copy(),
                    'target_action': np.asarray([expert.steering, applied[1]], dtype=np.float32),
                    'executed_action': executed.copy(), 'applied_action': applied.copy(),
                    'expert_action': expert.action.copy(), 'intervened': intervened,
                    'severity': severity, 'lateral_correction': lateral_delta,
                    'longitudinal_correction': longitudinal_delta,
                    'safety_intervention': bool(info.get('safety_intervention', False)),
                    'seed': int(seed), 'step': int(info['step']), 'stage': stage_name,
                    'collision': bool(info['collision']), 'out_of_road': bool(info['out_of_road']),
                    'success': bool(info['success']), 'truncated': bool(truncs['agent0']),
                    'route_completion': float(info['route_completion']),
                })
                steps += 1
                if terms['agent0'] or truncs['agent0']:
                    break
                observations = {'agent0': next_obs['agent0']}
        if not rows:
            raise RuntimeError('No privileged-expert transitions were collected')
        return rows
    finally:
        env.close()


def arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        'state': np.stack([row['state'] for row in rows]).astype(np.float32),
        'target_action': np.stack([row['target_action'] for row in rows]).astype(np.float32),
        'proposed_action': np.stack([row['proposed_action'] for row in rows]).astype(np.float32),
        'expert_action': np.stack([row['expert_action'] for row in rows]).astype(np.float32),
        'intervened': np.asarray([row['intervened'] for row in rows], dtype=bool),
        'severity': np.asarray([row['severity'] for row in rows], dtype=np.float32),
        'seed': np.asarray([row['seed'] for row in rows], dtype=np.int32),
        'step': np.asarray([row['step'] for row in rows], dtype=np.int32),
        'stage': np.asarray([row['stage'] for row in rows], dtype='U32'),
    }


def summary(rows: list[dict]) -> dict:
    lateral = np.asarray([row['lateral_correction'] for row in rows], dtype=np.float64)
    longitudinal = np.asarray([row['longitudinal_correction'] for row in rows], dtype=np.float64)
    terminal = [row for row in rows if row['success'] or row['collision'] or row['out_of_road'] or row['truncated']]
    return {
        'samples': len(rows), 'unique_seeds': len({row['seed'] for row in rows}),
        'stages': sorted({row['stage'] for row in rows}),
        'intervention_fraction': float(np.mean([row['intervened'] for row in rows])),
        'safety_intervention_fraction': float(np.mean([row['safety_intervention'] for row in rows])),
        'mean_lateral_correction': float(lateral.mean()),
        'p95_lateral_correction': float(np.quantile(lateral, .95)),
        'mean_longitudinal_correction': float(longitudinal.mean()),
        'terminal_episodes': len(terminal),
        'terminal_successes': int(sum(row['success'] for row in terminal)),
        'terminal_collisions': int(sum(row['collision'] for row in terminal)),
        'terminal_offroad': int(sum(row['out_of_road'] for row in terminal)),
        'terminal_timeouts': int(sum(row['truncated'] and not row['success'] for row in terminal)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--collection-config')
    p.add_argument('--curriculum-config', default=str(_bootstrap.ROOT / 'configs/v27_expert_curriculum.yaml'))
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--seeds-per-stage', type=int, default=4)
    p.add_argument('--max-steps-per-seed', type=int, default=260)
    p.add_argument('--epochs-per-stage', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--intervention-weight', type=float, default=3.0)
    p.add_argument('--mode', choices=['steering_only', 'balanced'], default='steering_only')
    p.add_argument('--expert-blend', type=float, default=1.0)
    p.add_argument('--ttc-threshold', type=float, default=1.8)
    p.add_argument('--risk-threshold', type=float, default=.68)
    args = p.parse_args()
    if min(args.seeds_per_stage, args.max_steps_per_seed, args.epochs_per_stage, args.batch_size) <= 0 or args.seed_offset < 0:
        p.error('count arguments must be positive and seed-offset nonnegative')
    if not 0 < args.expert_blend <= 1:
        p.error('--expert-blend must lie in (0,1]')

    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    model, source_config, payload = load_checkpoint(args.source)
    if source_config.env.backend != 'metadrive' or source_config.env.observation != 'state' or source_config.env.num_agents != 1:
        raise ValueError('Privileged expert distillation requires single-agent MetaDrive state PPO')
    base = copy.deepcopy(load_config(args.collection_config).env if args.collection_config else source_config.env)
    if base.backend != 'metadrive' or base.observation != 'state' or base.num_agents != 1:
        raise ValueError('Collection config must be single-agent MetaDrive state')
    base.safety_supervisor = True
    base.safety_ttc_threshold_s = args.ttc_threshold
    base.safety_risk_threshold = args.risk_threshold
    deploy_config = copy.deepcopy(source_config)
    deploy_config.env = copy.deepcopy(base)
    deploy_config.env.safety_supervisor = False
    stages = load_safety_curriculum(args.curriculum_config)
    save_json(runtime_manifest(), out / 'runtime.json')

    all_rows, evidence = [], []
    for index, stage in enumerate(stages):
        env_config = stage.apply(base)
        env_config.safety_supervisor = True
        seeds = expert_seed_block(args.seed_offset + index * args.seeds_per_stage, args.seeds_per_stage)
        rows = collect(model, env_config, seeds, args.max_steps_per_seed, stage.name, expert_blend=args.expert_blend)
        all_rows.extend(rows)
        data = arrays(all_rows)
        # Steering-only exactly preserves encoder + longitudinal actor. Balanced
        # mode also absorbs SafetySupervisor longitudinal targets and may alter
        # shared features; both are evaluated as raw students later.
        result = distill_safety_actor(
            model, data['state'], data['target_action'], data['intervened'], data['severity'],
            epochs=args.epochs_per_stage, batch_size=args.batch_size,
            learning_rate=args.learning_rate, intervention_weight=args.intervention_weight,
            steering_weight=1.0, train_encoder=(args.mode == 'balanced'),
            steering_only=(args.mode == 'steering_only'), device=source_config.device,
            seed=source_config.seed + index + 2700)
        stage_evidence = {
            'index': index, 'stage': stage.to_dict(), 'seeds': seeds,
            'new_collection': summary(rows), 'aggregate_collection': summary(all_rows),
            'distillation': result.to_dict(),
        }
        evidence.append(stage_evidence)
        save_checkpoint(out / f'stage_{index + 1}.pt', model, deploy_config, optimizer=None,
                        steps=int(payload.get('steps', 0)), extra={'privileged_expert_distillation': stage_evidence})

    data = arrays(all_rows)
    np.savez_compressed(out / 'expert_distillation_dataset.npz', **data)
    provenance = {
        'method': 'privileged_metadrive_idm_steering_distillation',
        'privilege_boundary': ('MetaDrive IDM teacher reads simulator-internal navigation/lidar only during collection; '
                               'raw student deployment receives the standard project state observation only.'),
        'source_model': str(Path(args.source).resolve()), 'source_sha256': sha256(args.source),
        'source_rl_world_steps': int(payload.get('steps', 0)), 'mode': args.mode,
        'expert_blend': float(args.expert_blend), 'expert_train_seed_offset': args.seed_offset,
        'curriculum': [stage.to_dict() for stage in stages], 'stages': evidence,
        'aggregate_collection': summary(all_rows),
        'critic_trained': False, 'log_std_trained': False,
        'deployment_uses_expert': False, 'deployment_safety_supervisor_enabled': False,
        'claim_boundary': 'privileged expert imitation is a training method, not a runtime safety controller or safety certificate',
    }
    final = out / 'model.pt'
    save_checkpoint(final, model, deploy_config, optimizer=None, steps=int(payload.get('steps', 0)),
                    extra={'privileged_expert_distillation': provenance})
    provenance['model_sha256'] = sha256(final)
    save_json(provenance, out / 'result.json')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
