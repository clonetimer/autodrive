"""Distill a frozen state-policy teacher into a raw-RGB+ego-motion actor.

Data are collected in the *real* MetaDrive environment.  The teacher receives
project state features built from the same physical snapshot; the student sees
only MetaDrive RGBCamera pixels plus ego/navigation motion.  Teacher actions are
executed during collection, so the dataset follows the teacher state
visitation.  This is behavior cloning, not an RGB RL performance claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import _bootstrap
from src.agents.distillation import distill_actor
from src.agents.trainer import load_checkpoint, model_from_config, save_checkpoint
from src.envs.metadrive_wrapper import MetaDriveParallelEnv
from src.envs.observation import build_state
from src.perception.encoder import stack_observations
from src.scenarios.generator import split_seeds
from src.utils.config import load_config, save_config
from src.utils.logger import save_json, sha256, runtime_manifest


def collect(teacher, teacher_cfg, student_cfg, samples: int, seed_count: int):
    env = MetaDriveParallelEnv(student_cfg.env)
    observations, actions, records = [], [], []
    seeds = split_seeds('train', seed_count)
    try:
        # Spread collection across distinct training seeds.  A single long
        # trajectory can make pixel/motion behavior cloning look deceptively
        # good in-sample while failing immediately after reset on a new road.
        per_seed = max(1, int(np.ceil(samples / seed_count)))
        candidate_seeds = split_seeds('train', min(200, max(seed_count, 1)))
        for seed in candidate_seeds:
            obs, _ = env.reset(seed=seed)
            collected_this_seed = 0
            while env.agents and len(actions) < samples and collected_this_seed < per_seed:
                snapshot = env._last_snapshots['agent0']
                teacher_obs = build_state(snapshot, teacher_cfg.env.neighbors)
                teacher_action = teacher.predict(teacher_obs, deterministic=True)
                observations.append({key: value.copy() for key, value in obs['agent0'].items()})
                actions.append(teacher_action.copy())
                next_obs, _, terms, truncs, infos = env.step({'agent0': teacher_action})
                info = infos['agent0']
                records.append({'seed': seed, 'step': int(info['step']), 'speed': float(info['speed']),
                                'route_completion': float(info['route_completion']),
                                'collision': bool(info['collision']), 'out_of_road': bool(info['out_of_road'])})
                collected_this_seed += 1
                if terms['agent0'] or truncs['agent0']:
                    break
                obs = {'agent0': next_obs['agent0']}
            if len(actions) >= samples:
                break
        if len(actions) < samples:
            # Continue with additional untouched training seeds, still limiting
            # each seed so the final dataset is not dominated by one episode.
            for seed in split_seeds('train', 200)[seed_count:]:
                obs, _ = env.reset(seed=seed)
                collected_this_seed = 0
                while env.agents and len(actions) < samples and collected_this_seed < per_seed:
                    snapshot = env._last_snapshots['agent0']
                    teacher_obs = build_state(snapshot, teacher_cfg.env.neighbors)
                    teacher_action = teacher.predict(teacher_obs, deterministic=True)
                    observations.append({key: value.copy() for key, value in obs['agent0'].items()})
                    actions.append(teacher_action.copy())
                    next_obs, _, terms, truncs, infos = env.step({'agent0': teacher_action})
                    info = infos['agent0']
                    records.append({'seed': seed, 'step': int(info['step']), 'speed': float(info['speed']),
                                    'route_completion': float(info['route_completion']),
                                    'collision': bool(info['collision']), 'out_of_road': bool(info['out_of_road'])})
                    collected_this_seed += 1
                    if terms['agent0'] or truncs['agent0']:
                        break
                    obs = {'agent0': next_obs['agent0']}
                if len(actions) >= samples:
                    break
        if len(actions) < samples:
            raise RuntimeError(f'Collected only {len(actions)} of {samples} requested teacher samples')
        return stack_observations(observations[:samples]), np.stack(actions[:samples]), records[:samples]
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', required=True)
    p.add_argument('--config', required=True, help='RGB student TrainConfig')
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=192)
    p.add_argument('--seed-count', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    args = p.parse_args()
    if args.samples <= 0 or args.seed_count <= 0:
        p.error('samples and seed-count must be positive')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    student_cfg = load_config(args.config)
    if student_cfg.env.backend != 'metadrive' or student_cfg.env.observation != 'rgb_ego':
        raise ValueError('Student config must use real MetaDrive rgb_ego observation')
    teacher, teacher_cfg, _ = load_checkpoint(args.teacher, student_cfg.device)
    if teacher_cfg.env.observation != 'state' or teacher_cfg.env.num_agents != 1:
        raise ValueError('Teacher must be a single-agent state policy')
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    save_config(student_cfg, out / 'config.yaml')
    save_json(runtime_manifest(), out / 'runtime.json')
    observations, targets, records = collect(teacher, teacher_cfg, student_cfg, args.samples, args.seed_count)
    np.savez_compressed(out / 'distillation_dataset.npz', **observations, target_action=targets)
    save_json(records, out / 'collection.json')
    student = model_from_config(student_cfg).to(student_cfg.device)
    result = distill_actor(student, observations, targets, epochs=args.epochs, batch_size=args.batch_size,
                           learning_rate=args.learning_rate, device=student_cfg.device, seed=student_cfg.seed)
    optimizer = torch.optim.Adam(list(student.encoder.parameters()) + list(student.actor.parameters()),
                                 lr=args.learning_rate)
    evidence = {'method': 'behavior_cloning_from_privileged_state_teacher',
                'teacher_sha256': sha256(args.teacher), 'teacher_observation': teacher_cfg.env.observation,
                'student_observation': student_cfg.env.observation, 'teacher_actions_executed_during_collection': True,
                'critic_trained': False, 'log_std_trained': False,
                'deployment_inputs': ['raw MetaDrive RGB', 'ego/navigation motion'], **result.to_dict()}
    save_checkpoint(out / 'model.pt', student, student_cfg, optimizer=None, steps=args.samples,
                    extra={'distillation': evidence})
    evidence['model_sha256'] = sha256(out / 'model.pt')
    evidence['status'] = 'distillation_completed_not_an_rgb_rl_or_safety_claim'
    save_json(evidence, out / 'result.json')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
