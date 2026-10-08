"""Train the v3.3 scene-aware temporal residual adapter with online DAgger.

The lateral teacher remains the transparent fixed-blend RouteAssist controller,
used only while collecting MetaDrive rollouts.  The deployed adapter sees only
policy-visible state history plus the frozen base action.  Training-time
SafetySupervisor labels optionally supervise a diagnostic risk head; that head
does not modify actions at inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import _bootstrap
from src.agents.trainer import load_checkpoint
from src.end_to_end.agent import EndToEndAgent
from src.evaluation.capability_suite import standard_capability_suite
from src.multi_agent.marl_env import make_parallel_env
from src.navigation.learned_route_adapter import (
    LearnedRouteAdapterAgent, load_route_adapter, policy_visible_curve_score)
from src.navigation.route_assist import RouteAssist
from src.navigation.scene_aware_adapter import (
    SceneAwareTemporalAdapter, SceneAwareTemporalAdapterConfig,
    save_scene_aware_adapter)
from src.scenarios.generator import split_seed_block
from src.utils.logger import save_json, sha256


STAGES = ('curve_tracking', 'ramp_merge')
DEFAULT_RISK_LOSS_WEIGHT = 0.0


def _specs(split: str, iteration: int, seeds_per_stage: int):
    template = {s.name: s for s in standard_capability_suite(split, seeds_per_stage)}
    block = split_seed_block(
        split, iteration * len(STAGES) * seeds_per_stage,
        len(STAGES) * seeds_per_stage)
    out = []
    for i, name in enumerate(STAGES):
        seeds = tuple(block[i * seeds_per_stage:(i + 1) * seeds_per_stage])
        out.append(replace(template[name], seeds=seeds))
    return tuple(out)


def _collect_worker(args):
    from src.evaluation.safety_suite import SafetySuiteSpec

    payload = json.loads(args.worker_spec_json)
    payload['seeds'] = tuple(payload['seeds'])
    spec = SafetySuiteSpec(**payload)
    base = EndToEndAgent(args.base_model)
    _, cfg, _ = load_checkpoint(args.base_model)
    env_cfg = spec.env(cfg.env)
    env_cfg.max_steps = min(env_cfg.max_steps, args.max_stage_steps)
    env_cfg.route_assist = False
    env_cfg.safety_supervisor = args.collection_safety == 'protected'
    env_cfg.safety_ttc_threshold_s = args.ttc_threshold
    env_cfg.safety_risk_threshold = args.risk_threshold
    env = make_parallel_env(env_cfg)
    learner = LearnedRouteAdapterAgent(args.base_model, args.adapter) if args.adapter else None
    teacher = RouteAssist(args.teacher_blend, mode='fixed')

    float_keys = (
        'state', 'base_action', 'learner_action', 'teacher_action',
        'teacher_steering', 'teacher_correction', 'executed_action',
        'applied_action', 'reward', 'safety_severity')
    bool_keys = ('safety_intervention', 'collision', 'out_of_road', 'success')
    rows = {k: [] for k in (*float_keys, *bool_keys, 'seed', 'step')}
    episodes = []
    try:
        for seed in spec.seeds:
            obs, _ = env.reset(seed=seed)
            if learner:
                learner.reset()
            success = collision = offroad = False
            steps = 0
            while env.agents:
                state = np.asarray(obs['agent0'], dtype=np.float32)
                base_action = base.predict(state)
                learner_action = learner.predict(state) if learner else base_action.copy()
                decision = teacher.decide(base_action, state)
                executed = learner_action.copy()
                executed[0] = np.clip(
                    args.mix_beta * decision.applied[0] +
                    (1.0 - args.mix_beta) * learner_action[0], -1.0, 1.0)

                nxt, rew, term, trunc, info = env.step({'agent0': executed})
                inf = info['agent0']

                rows['state'].append(state.copy())
                rows['base_action'].append(base_action.copy())
                rows['learner_action'].append(learner_action.copy())
                rows['teacher_action'].append(np.asarray(decision.applied, dtype=np.float32).copy())
                rows['teacher_steering'].append(float(decision.applied[0]))
                rows['teacher_correction'].append(float(decision.applied[0] - base_action[0]))
                rows['executed_action'].append(executed.copy())
                rows['applied_action'].append(np.asarray(inf['applied_action'], dtype=np.float32).copy())
                rows['reward'].append(float(rew['agent0']))
                rows['safety_intervention'].append(bool(inf.get('safety_intervention', False)))
                rows['safety_severity'].append(float(inf.get('safety_severity', 0.0)))
                rows['collision'].append(bool(inf['collision']))
                rows['out_of_road'].append(bool(inf['out_of_road']))
                rows['success'].append(bool(inf['success']))
                rows['seed'].append(int(seed))
                rows['step'].append(int(steps))

                steps += 1
                collision |= bool(inf['collision'])
                offroad |= bool(inf['out_of_road'])
                success |= bool(inf['success']) and bool(term['agent0'])
                obs = {n: nxt[n] for n in env.agents}
            episodes.append({
                'seed': int(seed), 'steps': steps,
                'success': bool(success and not collision and not offroad),
                'collision': collision, 'out_of_road': offroad,
            })

        arrays = {}
        for key, values in rows.items():
            if key in {'seed', 'step'}:
                dtype = np.int64
            elif key in bool_keys:
                dtype = np.bool_
            else:
                dtype = np.float32
            arrays[key] = np.asarray(values, dtype=dtype)
        out = Path(args.worker_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        save_json({
            'status': 'complete', 'stage': spec.name,
            'samples': len(arrays['state']), 'mix_beta': args.mix_beta,
            'teacher_blend': args.teacher_blend,
            'collection_safety': args.collection_safety,
            'safety_interventions': int(arrays['safety_intervention'].sum()),
            'episodes': episodes,
        }, out.with_suffix('.json'))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    finally:
        env.close()


def _concat(paths):
    parts = [np.load(p) for p in paths]
    keys = tuple(parts[0].files)
    for part in parts[1:]:
        if tuple(part.files) != keys:
            raise RuntimeError('DAgger dataset schemas do not match across collection parts')
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    for part in parts:
        part.close()
    return out


def _split_indices(seeds):
    unique = np.unique(seeds)
    if len(unique) < 2:
        raise RuntimeError('Need at least two scenario seeds')
    hold = set(unique[2::3].tolist() or unique[-1:].tolist())
    valid = np.asarray([i for i, x in enumerate(seeds) if int(x) in hold], dtype=np.int64)
    train = np.asarray([i for i, x in enumerate(seeds) if int(x) not in hold], dtype=np.int64)
    return train, valid, sorted(hold)


def _history_tensor(states: np.ndarray, seeds: np.ndarray, steps: np.ndarray,
                    history_len: int) -> np.ndarray:
    """Build left-padded histories without crossing episode/seed boundaries."""
    states = np.asarray(states, dtype=np.float32)
    seeds = np.asarray(seeds)
    steps = np.asarray(steps)
    if states.ndim != 2 or len(states) != len(seeds) or len(states) != len(steps):
        raise ValueError('Invalid transition arrays for temporal history construction')
    if history_len <= 0:
        raise ValueError('history_len must be positive')
    result = np.empty((len(states), history_len, states.shape[1]), dtype=np.float32)
    for seed in np.unique(seeds):
        idx = np.flatnonzero(seeds == seed)
        order = idx[np.argsort(steps[idx], kind='stable')]
        ordered_steps = steps[order]
        if len(np.unique(ordered_steps)) != len(ordered_steps):
            raise RuntimeError(f'Duplicate step index inside seed {int(seed)}')
        for position, target_idx in enumerate(order):
            start = max(0, position - history_len + 1)
            source = order[start:position + 1]
            missing = history_len - len(source)
            if missing:
                source = np.concatenate((np.repeat(source[:1], missing), source))
            result[target_idx] = states[source]
    return result


def _fit(model, arrays, epochs, lr, batch_size, seed, risk_loss_weight):
    train, valid, hold = _split_indices(arrays['seed'])
    histories = _history_tensor(
        arrays['state'], arrays['seed'], arrays['step'], model.config.history_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # A raw collection has no Supervisor labels by construction. Do not train a
    # constant-zero auxiliary task in that case, because its shared gradients
    # would bias the scene representation without carrying safety information.
    all_risk_targets = np.maximum(
        arrays['safety_severity'].astype(np.float32),
        arrays['safety_intervention'].astype(np.float32))
    effective_risk_weight = float(risk_loss_weight if np.any(all_risk_targets > 1e-6) else 0.0)
    model.train()

    for _ in range(epochs):
        perm = rng.permutation(train)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            hist = torch.as_tensor(histories[idx], dtype=torch.float32)
            ba = torch.as_tensor(arrays['base_action'][idx], dtype=torch.float32)
            target = torch.as_tensor(arrays['teacher_steering'][idx], dtype=torch.float32)
            risk_target = torch.as_tensor(
                np.maximum(arrays['safety_severity'][idx], arrays['safety_intervention'][idx].astype(np.float32)),
                dtype=torch.float32)
            pred, corr, risk_logit, _ = model(hist, ba)
            current = hist[:, -1]
            curve = policy_visible_curve_score(current)
            lane = current[:, 3].abs().clamp(0, 1)
            weight = 1.0 + 5.0 * curve + 2.0 * lane
            steering_loss = (weight * (pred - target).square()).mean() + 0.01 * corr.square().mean()
            risk_loss = F.binary_cross_entropy_with_logits(risk_logit, risk_target.clamp(0, 1))
            loss = steering_loss + effective_risk_weight * risk_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    model.eval()

    def metrics(idx):
        with torch.no_grad():
            hist = torch.as_tensor(histories[idx], dtype=torch.float32)
            ba = torch.as_tensor(arrays['base_action'][idx], dtype=torch.float32)
            target = torch.as_tensor(arrays['teacher_steering'][idx], dtype=torch.float32)
            risk_target = torch.as_tensor(
                np.maximum(arrays['safety_severity'][idx], arrays['safety_intervention'][idx].astype(np.float32)),
                dtype=torch.float32)
            pred, corr, risk_logit, attention = model(hist, ba)
            risk = torch.sigmoid(risk_logit)
            return {
                'samples': int(len(idx)),
                'steering_mse': float((pred - target).square().mean()),
                'steering_mae': float((pred - target).abs().mean()),
                'mean_abs_correction': float(corr.abs().mean()),
                'risk_brier': float((risk - risk_target).square().mean()),
                'risk_mean_prediction': float(risk.mean()),
                'risk_mean_target': float(risk_target.mean()),
                'neighbor_attention_mass': float(attention[:, :-1].sum(dim=1).mean()),
            }

    return {
        'train': metrics(train),
        'diagnostic_holdout': metrics(valid),
        'holdout_seeds': hold,
        'requested_risk_loss_weight': float(risk_loss_weight),
        'effective_risk_loss_weight': effective_risk_weight,
        'risk_supervision_available': bool(np.any(all_risk_targets > 1e-6)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', default='scene_train')
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--seeds-per-stage', type=int, default=2)
    p.add_argument('--training-seed', type=int, default=3300,
                   help='adapter initialization/minibatch seed; scenario seeds stay paired across trials')
    p.add_argument('--mix-betas', default='1.0,0.5,0.2')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--history-len', type=int, default=6)
    p.add_argument('--neighbor-heads', type=int, default=4)
    p.add_argument('--use-neighbors', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-correction', type=float, default=.18)
    p.add_argument('--risk-loss-weight', type=float, default=DEFAULT_RISK_LOSS_WEIGHT,
                   help='optional SafetySupervisor risk auxiliary weight; default is off and R15 is ablated separately')
    p.add_argument('--teacher-blend', type=float, default=.5)
    p.add_argument('--collection-safety', choices=['protected', 'raw'], default='protected')
    p.add_argument('--max-stage-steps', type=int, default=420)
    p.add_argument('--stage-timeout', type=float, default=120)
    p.add_argument('--ttc-threshold', type=float, default=1.8)
    p.add_argument('--risk-threshold', type=float, default=.68)
    p.add_argument('--resume', action='store_true',
                   help='resume fingerprint-compatible completed DAgger stages')
    p.add_argument('--worker-spec-json', help=argparse.SUPPRESS)
    p.add_argument('--worker-output', help=argparse.SUPPRESS)
    p.add_argument('--adapter', help=argparse.SUPPRESS)
    p.add_argument('--mix-beta', type=float, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker_spec_json:
        return _collect_worker(args)
    if args.risk_loss_weight < 0:
        p.error('risk-loss-weight must be nonnegative')
    betas = [float(x) for x in args.mix_betas.split(',')]
    if len(betas) != args.iterations:
        p.error('mix-betas count must equal iterations')

    _, cfg, _ = load_checkpoint(args.base_model)
    state_dim = 20 + 7 * cfg.env.neighbors
    torch.manual_seed(args.training_seed)
    model = SceneAwareTemporalAdapter(SceneAwareTemporalAdapterConfig(
        state_dim=state_dim, hidden_size=args.hidden_size,
        history_len=args.history_len, neighbor_heads=args.neighbor_heads,
        max_correction=args.max_correction, use_neighbors=args.use_neighbors))

    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError('Choose an empty output directory or use --resume')
    root.mkdir(parents=True, exist_ok=True)

    run_config = {
        'schema': 'v33_scene_aware_training_run_v1',
        'base_model_sha256': sha256(args.base_model),
        'split': args.split,
        'stages': list(STAGES),
        'iterations': args.iterations,
        'seeds_per_stage': args.seeds_per_stage,
        'mix_betas': betas,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'history_len': args.history_len,
        'neighbor_heads': args.neighbor_heads,
        'use_neighbors': bool(args.use_neighbors),
        'max_correction': args.max_correction,
        'risk_loss_weight': args.risk_loss_weight,
        'teacher_blend': args.teacher_blend,
        'collection_safety': args.collection_safety,
        'max_stage_steps': args.max_stage_steps,
        'ttc_threshold': args.ttc_threshold,
        'risk_threshold': args.risk_threshold,
        'training_seed': args.training_seed,
    }
    config_bytes = json.dumps(run_config, sort_keys=True, separators=(',', ':')).encode()
    run_config['fingerprint_sha256'] = hashlib.sha256(config_bytes).hexdigest()
    run_config_path = root / 'run_config.json'
    if args.resume:
        if not run_config_path.is_file():
            raise RuntimeError('Resume requires run_config.json from the original v3.3 run')
        existing = json.loads(run_config_path.read_text())
        if existing != run_config:
            raise RuntimeError('Resume configuration mismatch; use a new output directory')
    else:
        save_json(run_config, run_config_path)

    completed = []
    if args.resume and (root / 'iterations.json').is_file():
        completed = json.loads((root / 'iterations.json').read_text())
        if not isinstance(completed, list):
            raise RuntimeError('Invalid iterations.json')

    parts = []
    iterations = []
    adapter_path = None
    started = time.time()
    for it in range(args.iterations):
        idir = root / f'iteration_{it}'
        idir.mkdir(exist_ok=True)
        collection = []
        specs = _specs(args.split, it, args.seeds_per_stage)

        if it < len(completed):
            report = completed[it]
            if int(report.get('iteration', -1)) != it:
                raise RuntimeError('iterations.json ordering mismatch')
            for spec in specs:
                npz = idir / f'{spec.name}.npz'
                meta = npz.with_suffix('.json')
                if not npz.is_file() or not meta.is_file():
                    raise RuntimeError(f'Completed iteration {it} is missing {spec.name} data')
                summary = json.loads(meta.read_text())
                if summary.get('status') != 'complete' or summary.get('stage') != spec.name:
                    raise RuntimeError(f'Completed stage metadata mismatch: {meta}')
                parts.append(npz)
                collection.append(summary)
            adapter_path = Path(report['artifact']['path'])
            if not adapter_path.is_file():
                adapter_path = root / f'adapter_iteration_{it}.pt'
            model, _ = load_route_adapter(adapter_path, base_model=args.base_model, device='cpu')
            if not isinstance(model, SceneAwareTemporalAdapter):
                raise RuntimeError('Resume artifact is not a scene_aware_temporal_adapter_v3')
            iterations.append(report)
            continue

        for spec in specs:
            npz = idir / f'{spec.name}.npz'
            meta = npz.with_suffix('.json')
            log = npz.with_suffix('.log')
            reuse = False
            if args.resume and npz.is_file() and meta.is_file():
                try:
                    summary = json.loads(meta.read_text())
                    reuse = (
                        summary.get('status') == 'complete' and
                        summary.get('stage') == spec.name and
                        summary.get('collection_safety') == args.collection_safety)
                except (OSError, json.JSONDecodeError):
                    reuse = False
            if not reuse:
                npz.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    '--base-model', str(Path(args.base_model).resolve()),
                    '--output', str(root),
                    '--worker-spec-json', json.dumps(spec.to_dict()),
                    '--worker-output', str(npz),
                    '--mix-beta', str(betas[it]),
                    '--teacher-blend', str(args.teacher_blend),
                    '--collection-safety', args.collection_safety,
                    '--max-stage-steps', str(args.max_stage_steps),
                    '--ttc-threshold', str(args.ttc_threshold),
                    '--risk-threshold', str(args.risk_threshold),
                ]
                if adapter_path:
                    cmd += ['--adapter', str(adapter_path)]
                with log.open('w') as f:
                    proc = subprocess.run(
                        cmd, cwd=_bootstrap.ROOT, stdout=f, stderr=subprocess.STDOUT,
                        timeout=args.stage_timeout,
                        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                if proc.returncode or not npz.is_file():
                    raise RuntimeError(f'collection failed: {log}')
                summary = json.loads(meta.read_text())
            parts.append(npz)
            collection.append(summary)

        arrays = _concat(parts)
        fit = _fit(
            model, arrays, args.epochs, args.learning_rate, args.batch_size,
            args.training_seed + it, args.risk_loss_weight)
        adapter_path = root / f'adapter_iteration_{it}.pt'
        training = {
            'iteration': it, 'mix_beta': betas[it], 'fit': fit,
            'collection': collection, 'samples_accumulated': len(arrays['state']),
        }
        artifact = save_scene_aware_adapter(
            adapter_path, model, base_model=args.base_model,
            training=training, dataset=arrays)
        report = {**training, 'artifact': artifact}
        iterations.append(report)
        save_json(iterations, root / 'iterations.json')

    arrays = _concat(parts)
    final = root / 'scene_aware_temporal_adapter.pt'
    training = {
        'algorithm': 'online_dagger_scene_aware_temporal_adapter_v3',
        'iterations': iterations,
        'samples_total': len(arrays['state']),
        'unique_training_seeds': sorted(set(map(int, arrays['seed']))),
        'split': args.split,
        'stages': list(STAGES),
        'teacher': 'fixed_policy_visible_pure_pursuit',
        'teacher_blend': args.teacher_blend,
        'collection_safety': args.collection_safety,
        'risk_auxiliary_head': True,
        'risk_loss_weight': args.risk_loss_weight,
        'history_len': args.history_len,
        'use_neighbors': args.use_neighbors,
        'neighbor_heads': args.neighbor_heads,
        'base_model_sha256': sha256(args.base_model),
        'training_seed': args.training_seed,
        'run_config_fingerprint_sha256': run_config['fingerprint_sha256'],
        'elapsed_seconds': time.time() - started,
    }
    artifact = save_scene_aware_adapter(
        final, model, base_model=args.base_model, training=training, dataset=arrays)
    np.savez_compressed(root / 'dataset.npz', **arrays)
    result = {
        **training, 'artifact': artifact,
        'final_fit': iterations[-1]['fit'],
        'status': 'scene_aware_temporal_adapter_trained_not_safety_certified',
    }
    save_json(result, root / 'result.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
