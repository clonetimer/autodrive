"""Native PyTorch PPO / parameter-sharing IPPO / CTDE MAPPO training.

Collects real joint transitions; central state enters the critic ONLY. Supports
state, semantic-BEV and hybrid local actors, atomic checkpoints and CPU resume.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from torch import nn
from src.agents.policy import DrivingPolicy
from src.agents.rollout import generalized_advantage
from src.multi_agent.marl_env import make_parallel_env
from src.perception.encoder import stack_observations, tensor_observation, index_observation, zero_observation
from src.utils.config import TrainConfig, save_config
from src.utils.events import ScalarWriter
from src.utils.logger import append_csv, runtime_manifest, save_json, sha256
from src.utils.seed import set_seed
from src.scenarios.curriculum import Curriculum


def model_from_config(config: TrainConfig, central_dim: int = 0) -> DrivingPolicy:
    return DrivingPolicy(config.env.observation, config.env.neighbors, config.hidden_size, central_dim, config.env.bev_frames)


def save_checkpoint(path, model, config, optimizer=None, steps=0, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format_version': 2, 'config': config.to_dict(), 'central_dim': model.central_dim,
               'model': model.state_dict(), 'steps': int(steps), 'torch_rng': torch.get_rng_state(),
               'extra': extra or {}}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('format_version') != 2:
        raise ValueError('Not a supported format-version-2 checkpoint')
    # A historical checkpoint with no reward-version field belongs to v2;
    # never silently evaluate its return under the new default objective.
    payload['config']['env'].setdefault('reward_version', 2)
    payload['config']['env'].setdefault('bev_frames', 1)
    config = TrainConfig(**payload['config'])
    model = model_from_config(config, payload['central_dim']).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, config, payload


class PPOTrainer:
    def __init__(self, config: TrainConfig, output: str | Path, resume: str | Path | None = None):
        if config.algorithm == 'sac':
            raise ValueError('Use SACTrainer for off-policy training')
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / 'training.csv').exists() and resume is None:
            raise FileExistsError('Output already contains training logs; choose a new run directory or --resume')
        set_seed(config.seed, config.threads)
        self.curriculum = (Curriculum(config.curriculum_min_episodes, config.curriculum_success_threshold,
                                     config.curriculum_max_collision_rate) if config.curriculum else None)
        self.current_env_config = (replace(config.env, **self.curriculum.parameters())
                                   if self.curriculum else config.env)
        self.env = make_parallel_env(self.current_env_config)
        self.joint_outcomes = []
        self.names = self.env.possible_agents
        self.n_agents = len(self.names)
        central_dim = self.env.global_state_dim + self.n_agents if config.algorithm == 'mappo' else 0
        self.model = model_from_config(config, central_dim).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.reference_model = None
        self.reference_sha256 = None
        if config.reference_kl_coef > 0:
            reference, reference_cfg, _ = load_checkpoint(config.reference_checkpoint, config.device)
            architecture_reference = (reference_cfg.env.observation, reference_cfg.env.neighbors,
                                      reference_cfg.env.num_agents, reference_cfg.algorithm,
                                      reference_cfg.hidden_size, reference_cfg.env.bev_frames)
            architecture_target = (config.env.observation, config.env.neighbors, config.env.num_agents,
                                   config.algorithm, config.hidden_size, config.env.bev_frames)
            if architecture_reference != architecture_target or reference.central_dim != central_dim:
                self.env.close()
                raise ValueError('Reference policy architecture must match target actor/critic interface')
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            self.reference_model = reference
            self.reference_sha256 = sha256(config.reference_checkpoint)
        self.steps = 0
        if resume:
            loaded, previous, payload = load_checkpoint(resume, config.device)
            if (previous.env.observation, previous.env.neighbors, previous.env.num_agents, previous.algorithm,
                    previous.hidden_size, previous.env.bev_frames, previous.env.reward_version,
                    previous.env.reward_source, previous.env.rgb_width, previous.env.rgb_height) != (
                    config.env.observation, config.env.neighbors, config.env.num_agents,
                    config.algorithm, config.hidden_size, config.env.bev_frames, config.env.reward_version,
                    config.env.reward_source, config.env.rgb_width, config.env.rgb_height):
                self.env.close()
                raise ValueError('Resume observation, agent population or algorithm mismatch')
            self.model.load_state_dict(loaded.state_dict())
            self.optimizer.load_state_dict(payload['optimizer'])
            self.steps = payload['steps']
            torch.set_rng_state(payload['torch_rng'].cpu())
            if previous.curriculum != config.curriculum:
                self.env.close()
                raise ValueError('Resume must preserve the curriculum mode')
            if self.curriculum:
                state = payload.get('extra', {}).get('curriculum')
                if state is None:
                    self.env.close()
                    raise ValueError('Curriculum checkpoint is missing its stage/history')
                self.curriculum.stage = state['stage']
                self.curriculum.history.extend(tuple(row) for row in state['history'])
                self.env.close()
                self.current_env_config = replace(config.env, **self.curriculum.parameters())
                self.env = make_parallel_env(self.current_env_config)
        self.initial_parameters = {name: param.detach().cpu().clone() for name, param in self.model.named_parameters()}
        save_config(config, self.output / 'config.yaml')
        save_json(runtime_manifest(), self.output / 'runtime.json')
        self.writer = ScalarWriter(self.output / 'tensorboard')
        self.observations, _ = self.env.reset(seed=config.env.seed)
        self.template = zero_observation(next(iter(self.observations.values())))
        self.episodes = []
        self.agent_steps = 0

    def _batch(self, observations):
        return stack_observations([observations.get(name, self.template) for name in self.names])

    def _central(self, state, truncated=None):
        if not self.model.central_dim:
            return None
        rows = np.tile(state, (self.n_agents, 1))
        if truncated is not None and hasattr(self.env, 'transition_state'):
            for i, flag in enumerate(truncated):
                if flag:
                    rows[i] = self.env.transition_state
        return np.concatenate((rows, np.eye(self.n_agents, dtype=np.float32)), axis=1).astype(np.float32)

    def _collect(self, count):
        cfg = self.config
        history = {key: [] for key in ['obs', 'central', 'latent', 'log_prob', 'value', 'next_value',
                                          'reward', 'terminated', 'truncated', 'valid']}
        self.model.eval()
        for _ in range(count):
            active = self.env.agents.copy()
            observations = self._batch(self.observations)
            central = self._central(self.env.global_state())
            obs_tensor = tensor_observation(observations, cfg.device)
            central_tensor = torch.as_tensor(central, device=cfg.device) if central is not None else None
            with torch.no_grad():
                action, latent, log_prob, value = self.model.sample(obs_tensor, central_tensor)
            actions = {name: action[i].cpu().numpy() for i, name in enumerate(self.names) if name in active}
            next_obs, rewards, terms, truncs, infos = self.env.step(actions)
            term_array = np.array([terms.get(name, True) for name in self.names], dtype=bool)
            trunc_array = np.array([truncs.get(name, False) for name in self.names], dtype=bool)
            next_central = self._central(self.env.global_state(), trunc_array)
            with torch.no_grad():
                next_value = self.model.value(tensor_observation(self._batch(next_obs), cfg.device),
                    torch.as_tensor(next_central, device=cfg.device) if next_central is not None else None)
            history['obs'].append(observations)
            if central is not None:
                history['central'].append(central)
            history['latent'].append(latent.cpu().numpy())
            history['log_prob'].append(log_prob.cpu().numpy())
            history['value'].append(value.cpu().numpy())
            history['next_value'].append(next_value.cpu().numpy())
            history['reward'].append([rewards.get(name, 0.0) for name in self.names])
            history['terminated'].append(term_array)
            history['truncated'].append(trunc_array)
            history['valid'].append([name in active for name in self.names])
            self.steps += 1
            self.agent_steps += len(active)
            for name in active:
                if terms[name] or truncs[name]:
                    info = infos[name]
                    episode = {'world_step': self.steps, 'agent': name,
                               'return': info.get('episode_return', 0.0), 'success': info['success'],
                               'collision': info['collision'], 'out_of_road': info['out_of_road'],
                               'truncated': truncs[name], 'completion': info['route_completion']}
                    self.episodes.append(episode)
                    self.joint_outcomes.append((episode['success'], episode['collision']))
                    append_csv(self.output / 'episodes.csv', episode)
            if not self.env.agents:
                changed = self._record_joint_outcome()
                self.observations, _ = (self.env.reset(seed=cfg.env.seed + self.curriculum.stage * 100)
                                        if changed else self.env.reset())
            else:
                self.observations = {name: next_obs[name] for name in self.env.agents}
        return history

    def _record_joint_outcome(self) -> bool:
        """Called only at a completed joint episode; no partial-agent promotion."""
        changed = False
        if self.curriculum:
            old_stage = self.curriculum.stage
            changed = self.curriculum.observe(all(x[0] for x in self.joint_outcomes),
                                             any(x[1] for x in self.joint_outcomes))
            append_csv(self.output / 'curriculum.csv', {
                'world_step': self.steps, 'stage_before': old_stage, 'stage_after': self.curriculum.stage,
                'joint_success': all(x[0] for x in self.joint_outcomes),
                'any_collision': any(x[1] for x in self.joint_outcomes), 'advanced': changed})
            if changed:
                self.env.close()
                self.current_env_config = replace(self.config.env, **self.curriculum.parameters())
                self.env = make_parallel_env(self.current_env_config)
        self.joint_outcomes.clear()
        return changed

    def _checkpoint_extra(self):
        return {'curriculum': {'stage': self.curriculum.stage, 'history': list(self.curriculum.history)}} if self.curriculum else {}

    def _update(self, data):
        cfg = self.config
        advantage, returns = generalized_advantage(data['reward'], data['value'], data['next_value'],
            data['terminated'], data['truncated'], data['valid'], cfg.gamma, cfg.gae_lambda)
        valid = np.asarray(data['valid'], dtype=bool).reshape(-1)
        obs = stack_observations(data['obs'])
        if isinstance(obs, dict):
            obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in obs.items()}
        else:
            obs = obs.reshape((-1,) + obs.shape[2:])
        obs = tensor_observation(obs, cfg.device)
        tensors = {key: torch.as_tensor(np.asarray(data[key]).reshape(-1, 2) if key == 'latent'
                                        else np.asarray(data[key]).reshape(-1), device=cfg.device, dtype=torch.float32)
                   for key in ['latent', 'log_prob', 'value']}
        central = torch.as_tensor(np.asarray(data['central']).reshape(-1, self.model.central_dim), device=cfg.device) if self.model.central_dim else None
        adv = torch.as_tensor(advantage.reshape(-1), device=cfg.device)
        ret = torch.as_tensor(returns.reshape(-1), device=cfg.device)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise RuntimeError('No valid transitions were collected')
        adv_mean, adv_std = adv[indices].mean(), adv[indices].std(unbiased=False)
        adv = (adv - adv_mean) / (adv_std + 1e-8)
        metrics = []
        self.model.train()
        stop = False
        for _ in range(cfg.epochs):
            for batch in np.array_split(np.random.permutation(indices), max(1, math_ceil(len(indices), cfg.batch_size))):
                new_log_prob, entropy, value = self.model.evaluate(index_observation(obs, batch), tensors['latent'][batch],
                    central[batch] if central is not None else None)
                log_ratio = new_log_prob - tensors['log_prob'][batch]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(adv[batch] * ratio,
                    adv[batch] * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
                clipped_value = tensors['value'][batch] + (value - tensors['value'][batch]).clamp(-cfg.clip_range, cfg.clip_range)
                value_loss = 0.5 * torch.maximum((value - ret[batch]) ** 2, (clipped_value - ret[batch]) ** 2).mean()
                reference_kl = torch.zeros((), device=cfg.device)
                if self.reference_model is not None:
                    batch_obs = index_observation(obs, batch)
                    current_distribution, _ = self.model.distribution(batch_obs)
                    with torch.no_grad():
                        reference_distribution, _ = self.reference_model.distribution(batch_obs)
                    reference_kl = torch.distributions.kl_divergence(
                        current_distribution, reference_distribution).sum(-1).mean()
                loss = (policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
                        + cfg.reference_kl_coef * reference_kl)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss; run aborted without publishing a success record')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError('Non-finite gradients')
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                metrics.append([loss.item(), policy_loss.item(), value_loss.item(), entropy.mean().item(), kl, clip_fraction, gradient_norm.item(), reference_kl.item()])
                if kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        names = ['loss', 'policy_loss', 'value_loss', 'gaussian_entropy', 'approx_kl', 'clip_fraction', 'gradient_norm', 'reference_kl']
        result = dict(zip(names, np.mean(metrics, axis=0).tolist()))
        valid_returns = returns.reshape(-1)[valid]
        residual = valid_returns - np.asarray(data['value']).reshape(-1)[valid]
        variance = float(np.var(valid_returns))
        result['explained_variance'] = float(1 - np.var(residual) / variance) if variance > 1e-10 else 0.0
        result['explained_variance_defined'] = int(variance > 1e-10)
        actions = np.tanh(np.asarray(data['latent']).reshape(-1, 2)[valid])
        result['action_saturation_fraction'] = float((np.abs(actions) > .95).mean())
        return result

    def train(self) -> dict:
        start = time.perf_counter()
        start_steps = self.steps
        last_checkpoint = self.steps
        try:
            while self.steps < self.config.total_steps:
                data = self._collect(min(self.config.rollout_steps, self.config.total_steps - self.steps))
                metrics = self._update(data)
                elapsed = time.perf_counter() - start
                recent = self.episodes[-30:]
                row = {'world_steps': self.steps, 'active_agent_steps_this_run': self.agent_steps,
                       'elapsed_seconds': elapsed, 'world_steps_per_second': (self.steps - start_steps) / max(elapsed, 1e-9),
                       'episodes': len(self.episodes), 'mean_recent_return': float(np.mean([x['return'] for x in recent])) if recent else 0.0,
                       **metrics}
                append_csv(self.output / 'training.csv', row)
                for key, value in row.items():
                    if key != 'world_steps':
                        self.writer.add_scalar('train/' + key, value, self.steps)
                if self.steps - last_checkpoint >= self.config.checkpoint_every:
                    save_checkpoint(self.output / 'checkpoints' / f'step_{self.steps}.pt', self.model, self.config,
                                    self.optimizer, self.steps, extra=self._checkpoint_extra())
                    last_checkpoint = self.steps
            final = self.output / 'model.pt'
            save_checkpoint(final, self.model, self.config, self.optimizer, self.steps, extra=self._checkpoint_extra())
            changes = {name: float((param.detach().cpu() - self.initial_parameters[name]).abs().sum())
                       for name, param in self.model.named_parameters()}
            result = {'backend': self.config.env.backend, 'algorithm': self.config.algorithm,
                'observation': self.config.env.observation, 'num_agents': self.n_agents,
                'world_steps_total': self.steps, 'world_steps_this_run': self.steps - start_steps,
                'active_agent_steps_this_run': self.agent_steps, 'elapsed_seconds': time.perf_counter() - start,
                'episodes_finished': len(self.episodes), 'model_sha256': sha256(final),
                'parameters_changed_l1': float(sum(changes.values())),
                'encoder_changed_l1': float(sum(value for key, value in changes.items() if key.startswith('encoder.'))),
                'resume_semantics': 'optimizer/weights/counter restored; world restarts at an episode boundary, not bitwise trajectory continuation',
                'curriculum_final_stage': self.curriculum.stage if self.curriculum else None,
                'reference_policy_sha256': self.reference_sha256,
                'reference_kl_coef': self.config.reference_kl_coef,
                'status': 'training_completed_not_a_driving_safety_claim'}
            save_json(result, self.output / 'result.json')
            return result
        finally:
            self.writer.close()
            self.env.close()


def math_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def train(config: TrainConfig, output, resume=None):
    if config.algorithm == 'sac':
        from src.agents.sac_agent import SACTrainer
        return SACTrainer(config, output, resume).train()
    return PPOTrainer(config, output, resume).train()
