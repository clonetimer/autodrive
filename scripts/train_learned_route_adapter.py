"""Online DAgger distillation of transparent RouteAssist into a learned residual adapter.

Fresh MetaDrive states are gathered under a mixture of the current learned
adapter and a configurable transparent RouteAssist teacher.  Only policy-visible
state features are stored.  The runtime artifact later predicts steering
correction/blend with a neural network; it does not call pure pursuit.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn

import _bootstrap
from src.agents.trainer import load_checkpoint
from src.end_to_end.agent import EndToEndAgent
from src.evaluation.capability_suite import standard_capability_suite
from src.evaluation.metrics import summarize
from src.multi_agent.marl_env import make_parallel_env
from src.navigation.learned_route_adapter import (
    LearnedRouteAdapter, LearnedRouteAdapterAgent, RouteAdapterConfig, save_route_adapter,
    policy_visible_curve_score,
)
from src.navigation.route_assist import RouteAssist
from src.scenarios.generator import split_seed_block
from src.utils.logger import save_json, sha256


def _suite_for_iteration(split: str, iteration: int, seeds_per_stage: int):
    count = 5 * seeds_per_stage
    block = split_seed_block(split, iteration * count, count)
    template = standard_capability_suite(split, seeds_per_stage)
    result = []
    for index, spec in enumerate(template):
        seeds = tuple(block[index * seeds_per_stage:(index + 1) * seeds_per_stage])
        result.append(replace(spec, seeds=seeds))
    return tuple(result)


def _worker(args) -> int:
    spec_dict = json.loads(args.worker_spec_json)
    from src.evaluation.safety_suite import SafetySuiteSpec
    spec_dict['seeds'] = tuple(spec_dict['seeds'])
    spec = SafetySuiteSpec(**spec_dict)
    base = EndToEndAgent(args.base_model)
    _, cfg, _ = load_checkpoint(args.base_model)
    env_cfg = spec.env(cfg.env)
    if args.max_stage_steps > 0:
        env_cfg.max_steps = min(env_cfg.max_steps, int(args.max_stage_steps))
    env_cfg.route_assist = False
    env_cfg.safety_supervisor = bool(args.supervised_collection)
    env_cfg.safety_ttc_threshold_s = args.ttc_threshold
    env_cfg.safety_risk_threshold = args.risk_threshold
    env = make_parallel_env(env_cfg)
    adapter_agent = (LearnedRouteAdapterAgent(args.base_model, args.adapter)
                     if args.adapter else None)
    teacher_mode = args.teacher_mode
    if teacher_mode == 'curve_fixed':
        teacher_mode = 'fixed' if spec.name in {'curve_tracking', 'ramp_merge'} else 'adaptive'
    teacher = RouteAssist(
        args.teacher_max_blend, mode=teacher_mode, min_blend=0.0,
        curvature_start=args.curvature_start, curvature_full=args.curvature_full,
        heading_start_rad=args.heading_start_rad, heading_full_rad=args.heading_full_rad,
        lane_start=args.lane_start, lane_full=args.lane_full,
    )
    rows = {key: [] for key in ['state', 'base_action', 'learner_action', 'teacher_steering',
                                 'teacher_raw_delta', 'teacher_blend', 'teacher_need',
                                 'executed_steering', 'seed']}
    episode_rows = []
    try:
        for seed in spec.seeds:
            obs, _ = env.reset(seed=seed)
            if adapter_agent is not None:
                adapter_agent.reset()
            total_return, steps = 0.0, 0
            collision = offroad = success = False
            while env.agents:
                state = np.asarray(obs['agent0'], dtype=np.float32)
                base_action = base.predict(state)
                learner_action = (adapter_agent.predict(state) if adapter_agent is not None else base_action.copy())
                decision = teacher.decide(base_action, state)
                executed = learner_action.copy()
                executed[0] = np.clip(
                    args.mix_beta * float(decision.applied[0]) +
                    (1.0 - args.mix_beta) * float(learner_action[0]), -1.0, 1.0)
                rows['state'].append(state.copy())
                rows['base_action'].append(base_action.copy())
                rows['learner_action'].append(learner_action.copy())
                rows['teacher_steering'].append(float(decision.applied[0]))
                rows['teacher_raw_delta'].append(float(decision.target_steering - base_action[0]))
                rows['teacher_blend'].append(float(decision.blend))
                rows['teacher_need'].append(float(decision.need_score))
                rows['executed_steering'].append(float(executed[0]))
                rows['seed'].append(int(seed))
                next_obs, rewards, terms, truncs, infos = env.step({'agent0': executed})
                info = infos['agent0']
                total_return += float(rewards['agent0'])
                steps += 1
                collision |= bool(info['collision'])
                offroad |= bool(info['out_of_road'])
                success |= bool(info['success']) and bool(terms['agent0'])
                obs = {name: next_obs[name] for name in env.agents}
            episode_rows.append({'seed': int(seed), 'steps': steps, 'return': total_return,
                                 'success': bool(success and not collision and not offroad),
                                 'collision': collision, 'out_of_road': offroad})
        arrays = {
            'state': np.asarray(rows['state'], dtype=np.float32),
            'base_action': np.asarray(rows['base_action'], dtype=np.float32),
            'learner_action': np.asarray(rows['learner_action'], dtype=np.float32),
            'teacher_steering': np.asarray(rows['teacher_steering'], dtype=np.float32),
            'teacher_raw_delta': np.asarray(rows['teacher_raw_delta'], dtype=np.float32),
            'teacher_blend': np.asarray(rows['teacher_blend'], dtype=np.float32),
            'teacher_need': np.asarray(rows['teacher_need'], dtype=np.float32),
            'executed_steering': np.asarray(rows['executed_steering'], dtype=np.float32),
            'seed': np.asarray(rows['seed'], dtype=np.int64),
        }
        out = Path(args.worker_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        save_json({'status': 'complete', 'stage': spec.name, 'spec': spec.to_dict(),
                   'samples': len(arrays['state']), 'mix_beta': args.mix_beta,
                   'supervised_collection': bool(args.supervised_collection),
                   'teacher_mode_requested': args.teacher_mode, 'teacher_mode_effective': teacher_mode,
                   'episodes': episode_rows}, out.with_suffix('.json'))
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    finally:
        env.close()


def _concat(parts: list[Path]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in parts]
    keys = loaded[0].files
    result = {key: np.concatenate([part[key] for part in loaded], axis=0) for key in keys}
    for part in loaded:
        part.close()
    return result


def _loss(model, state, base_action, target_steer, target_raw, target_blend, curve_loss_weight: float = 0.0):
    pred_steer, pred_raw, pred_blend = model(state, base_action)
    normalized_blend = target_blend / max(model.config.max_blend, 1e-6)
    lane = state[:, 3].abs().clamp(0, 1)
    curve = policy_visible_curve_score(state)
    weight = 1.0 + 4.0 * normalized_blend + 1.5 * lane + float(curve_loss_weight) * curve
    final = (weight * (pred_steer - target_steer).square()).mean()
    raw = (weight * (pred_raw - target_raw).square()).mean()
    blend = (weight * (pred_blend - target_blend).square()).mean()
    return final + 0.08 * raw + 0.25 * blend, final, raw, blend


def _evaluate_fit(model, arrays, indices, device):
    if len(indices) == 0:
        return {}
    with torch.no_grad():
        st = torch.as_tensor(arrays['state'][indices], dtype=torch.float32, device=device)
        ba = torch.as_tensor(arrays['base_action'][indices], dtype=torch.float32, device=device)
        target = torch.as_tensor(arrays['teacher_steering'][indices], dtype=torch.float32, device=device)
        target_raw = torch.as_tensor(arrays['teacher_raw_delta'][indices], dtype=torch.float32, device=device)
        target_blend = torch.as_tensor(arrays['teacher_blend'][indices], dtype=torch.float32, device=device)
        pred, raw, blend = model(st, ba)
        return {
            'samples': int(len(indices)),
            'steering_mse': float((pred-target).square().mean().cpu()),
            'steering_mae': float((pred-target).abs().mean().cpu()),
            'blend_mse': float((blend-target_blend).square().mean().cpu()),
            'raw_delta_mse': float((raw-target_raw).square().mean().cpu()),
            'blend_mean_predicted': float(blend.mean().cpu()),
            'blend_mean_teacher': float(target_blend.mean().cpu()),
        }


def _train_adapter(model, arrays, *, epochs, batch_size, learning_rate, seed, device, curve_loss_weight=0.0):
    seeds = arrays['seed']
    unique = np.unique(seeds)
    # Scenario-level split. Historical all-capability runs used every fifth
    # seed. Focused v3.2 curricula can begin with only two stage seeds in the
    # first DAgger iteration, so reserve the last scenario until the dataset
    # grows large enough for the historical 20% cadence.
    if len(unique) < 2:
        raise RuntimeError('Need at least two scenario seeds for adapter train/diagnostic split')
    holdout_values = unique[4::5] if len(unique) >= 5 else unique[-1:]
    holdout_seeds = set(holdout_values.tolist())
    valid = np.array([i for i, seed_value in enumerate(seeds) if int(seed_value) in holdout_seeds], dtype=np.int64)
    train = np.array([i for i, seed_value in enumerate(seeds) if int(seed_value) not in holdout_seeds], dtype=np.int64)
    if not len(train) or not len(valid):
        raise RuntimeError('Scenario-level adapter split produced an empty partition')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    history = []
    for epoch in range(epochs):
        permutation = rng.permutation(train)
        losses = []
        for start in range(0, len(permutation), batch_size):
            idx = permutation[start:start+batch_size]
            st = torch.as_tensor(arrays['state'][idx], dtype=torch.float32, device=device)
            ba = torch.as_tensor(arrays['base_action'][idx], dtype=torch.float32, device=device)
            ts = torch.as_tensor(arrays['teacher_steering'][idx], dtype=torch.float32, device=device)
            tr = torch.as_tensor(arrays['teacher_raw_delta'][idx], dtype=torch.float32, device=device)
            tb = torch.as_tensor(arrays['teacher_blend'][idx], dtype=torch.float32, device=device)
            loss, final, raw, blend = _loss(model, st, ba, ts, tr, tb, curve_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite route-adapter loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append([float(loss.detach()), float(final.detach()), float(raw.detach()), float(blend.detach())])
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses, axis=0)[0])})
    model.eval()
    return {'train': _evaluate_fit(model, arrays, train, device),
            'diagnostic_holdout': _evaluate_fit(model, arrays, valid, device),
            'holdout_seeds': sorted(holdout_seeds), 'epochs': epochs,
            'last_epoch_loss': history[-1]['loss']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', default='adapter_train')
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--seeds-per-stage', type=int, default=2)
    p.add_argument('--mix-betas', default='1.0,0.5,0.2')
    p.add_argument('--epochs', type=int, default=18)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=8e-4)
    p.add_argument('--hidden-size', type=int, default=96)
    p.add_argument('--teacher-max-blend', type=float, default=0.5)
    p.add_argument('--teacher-mode', choices=['adaptive', 'fixed', 'curve_fixed'], default='adaptive',
                   help='curve_fixed uses fixed assistance on curve/ramp stages and adaptive elsewhere')
    p.add_argument('--curve-loss-weight', type=float, default=0.0,
                   help='extra imitation weight for policy-visible curved route previews')
    p.add_argument('--supervised-collection', action='store_true', default=True)
    p.add_argument('--ttc-threshold', type=float, default=1.8)
    p.add_argument('--risk-threshold', type=float, default=.68)
    p.add_argument('--worker-spec-json', help=argparse.SUPPRESS)
    p.add_argument('--worker-output', help=argparse.SUPPRESS)
    p.add_argument('--adapter', help=argparse.SUPPRESS)
    p.add_argument('--mix-beta', type=float, help=argparse.SUPPRESS)
    p.add_argument('--curvature-start', type=float, default=.06)
    p.add_argument('--curvature-full', type=float, default=.28)
    p.add_argument('--heading-start-rad', type=float, default=.035)
    p.add_argument('--heading-full-rad', type=float, default=.18)
    p.add_argument('--lane-start', type=float, default=.06)
    p.add_argument('--lane-full', type=float, default=.30)
    p.add_argument('--stage-timeout', type=float, default=180.0)
    p.add_argument('--focus-stages', default='',
                   help='comma-separated capability stages to collect; empty keeps all five')
    p.add_argument('--max-stage-steps', type=int, default=320, help='cap collection horizon to keep DAgger workers bounded')
    args = p.parse_args()
    if args.worker_spec_json:
        return _worker(args)
    if args.iterations <= 0 or args.seeds_per_stage <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        p.error('iterations/seeds/epochs/batch-size must be positive')
    betas = [float(x) for x in args.mix_betas.split(',') if x.strip()]
    if len(betas) != args.iterations or any(not 0 <= x <= 1 for x in betas):
        p.error('--mix-betas must contain one [0,1] value per iteration')
    base_policy, cfg, _ = load_checkpoint(args.base_model)
    if cfg.env.backend != 'metadrive' or cfg.env.observation != 'state' or cfg.env.num_agents != 1:
        p.error('Learned route adapter requires a single-agent MetaDrive state checkpoint')
    state_dim = int(cfg.env.neighbors * 7 + 20)
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    model = LearnedRouteAdapter(RouteAdapterConfig(state_dim, args.hidden_size, args.teacher_max_blend)).cpu()
    parts: list[Path] = []
    iteration_reports = []
    adapter_path: Path | None = None
    started = time.time()
    for iteration in range(args.iterations):
        suite = _suite_for_iteration(args.split, iteration, args.seeds_per_stage)
        if args.focus_stages:
            requested = {item.strip() for item in args.focus_stages.split(',') if item.strip()}
            known = {spec.name for spec in suite}
            unknown = requested - known
            if unknown:
                raise ValueError(f'Unknown focus stages: {sorted(unknown)}')
            suite = tuple(spec for spec in suite if spec.name in requested)
            if not suite:
                raise ValueError('focus-stages selected no capability stages')
        iteration_dir = root / f'iteration_{iteration}'
        iteration_dir.mkdir()
        worker_summaries = []
        for spec in suite:
            npz = iteration_dir / f'{spec.name}.npz'
            cmd = [sys.executable, str(Path(__file__).resolve()), '--base-model', str(Path(args.base_model).resolve()),
                   '--output', str(root), '--worker-spec-json', json.dumps(spec.to_dict()),
                   '--worker-output', str(npz), '--mix-beta', str(betas[iteration]),
                   '--teacher-max-blend', str(args.teacher_max_blend), '--teacher-mode', args.teacher_mode, '--ttc-threshold', str(args.ttc_threshold),
                   '--risk-threshold', str(args.risk_threshold), '--curvature-start', str(args.curvature_start),
                   '--curvature-full', str(args.curvature_full), '--heading-start-rad', str(args.heading_start_rad),
                   '--heading-full-rad', str(args.heading_full_rad), '--lane-start', str(args.lane_start),
                   '--lane-full', str(args.lane_full), '--max-stage-steps', str(args.max_stage_steps)]
            if args.supervised_collection:
                cmd.append('--supervised-collection')
            if adapter_path is not None:
                cmd += ['--adapter', str(adapter_path)]
            log = npz.with_suffix('.log')
            before = time.time()
            with log.open('w', encoding='utf-8') as handle:
                process = subprocess.run(cmd, cwd=_bootstrap.ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                         timeout=args.stage_timeout, env={**os.environ, 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            if process.returncode != 0 or not npz.is_file():
                raise RuntimeError(f'Collection stage {spec.name} failed; inspect {log}')
            parts.append(npz)
            summary = json.loads(npz.with_suffix('.json').read_text())
            worker_summaries.append({**summary, 'elapsed_seconds': time.time()-before})
        arrays = _concat(parts)
        fit = _train_adapter(model, arrays, epochs=args.epochs, batch_size=args.batch_size,
                             learning_rate=args.learning_rate, seed=3100+iteration, device='cpu',
                             curve_loss_weight=args.curve_loss_weight)
        adapter_path = root / f'adapter_iteration_{iteration}.pt'
        training = {'iteration': iteration, 'mix_beta': betas[iteration], 'fit': fit,
                    'collection': worker_summaries, 'samples_accumulated': int(len(arrays['state']))}
        artifact = save_route_adapter(adapter_path, model, base_model=args.base_model, training=training, dataset=arrays)
        iteration_reports.append({**training, 'artifact': artifact})
        save_json(iteration_reports, root / 'iterations.json')
    arrays = _concat(parts)
    final_path = root / 'route_adapter.pt'
    final_training = {'algorithm': 'online_dagger_route_adapter', 'iterations': iteration_reports,
                      'samples_total': int(len(arrays['state'])),
                      'unique_training_seeds': sorted(set(int(x) for x in arrays['seed'])),
                      'split': args.split, 'teacher': f'{args.teacher_mode}_policy_visible_pure_pursuit',
                      'teacher_mode': args.teacher_mode, 'teacher_max_blend': args.teacher_max_blend,
                      'curve_loss_weight': args.curve_loss_weight,
                      'focus_stages': [x.strip() for x in args.focus_stages.split(',') if x.strip()],
                      'supervised_collection': bool(args.supervised_collection),
                      'base_model_sha256': sha256(args.base_model),
                      'elapsed_seconds': time.time()-started}
    artifact = save_route_adapter(final_path, model, base_model=args.base_model, training=final_training, dataset=arrays)
    np.savez_compressed(root / 'dataset.npz', **arrays)
    result = {**final_training, 'artifact': artifact,
              'final_fit': iteration_reports[-1]['fit'],
              'status': 'learned_route_adapter_trained_not_a_safety_claim'}
    save_json(result, root / 'result.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
