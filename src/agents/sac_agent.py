"""Actual CPU SAC: twin Q, replay, target networks and learned temperature.

Single learning ego only; background IDM vehicles remain interactive. Unlike
PPO this uses an off-policy replay buffer. Pixel observations keep uint8 in RAM.
"""
from __future__ import annotations
from pathlib import Path
import copy
import math
import time
import random
import numpy as np
import torch
from torch import nn
from src.agents.policy import DrivingPolicy
from src.agents.trainer import model_from_config, save_checkpoint, load_checkpoint
from src.multi_agent.marl_env import make_parallel_env
from src.perception.encoder import ObservationEncoder, stack_observations, tensor_observation
from src.utils.config import save_config
from src.utils.events import ScalarWriter
from src.utils.logger import append_csv, runtime_manifest, save_json, sha256
from src.utils.seed import set_seed


class ReplayBuffer:
    def __init__(self, capacity: int, example, seed: int, max_bytes: int = 536870912):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.dictionary = isinstance(example, dict)
        example = example if self.dictionary else {'state': example}
        required = capacity * (2 * sum(value.nbytes for value in example.values()) + 16)
        if capacity <= 0 or required > max_bytes:
            raise ValueError(f'Replay requires {required} bytes; limit is {max_bytes}. Reduce replay_size or image dimensions.')
        self.obs = {key: np.zeros((capacity,) + value.shape, dtype=value.dtype) for key, value in example.items()}
        self.next_obs = {key: np.zeros_like(value) for key, value in self.obs.items()}
        self.actions = np.zeros((capacity, 2), np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)

    def add(self, obs, action, reward, next_obs, terminated):
        obs = obs if self.dictionary else {'state': obs}
        next_obs = next_obs if self.dictionary else {'state': next_obs}
        index = self.position
        for key in self.obs:
            self.obs[key][index] = obs[key]
            self.next_obs[key][index] = next_obs[key]
        self.actions[index], self.rewards[index], self.terminated[index] = action, reward, terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, device='cpu'):
        if self.size == 0:
            raise ValueError('Cannot sample an empty replay buffer')
        index = self.rng.integers(self.size, size=batch_size)
        obs = {key: torch.as_tensor(value[index], device=device) for key, value in self.obs.items()}
        next_obs = {key: torch.as_tensor(value[index], device=device) for key, value in self.next_obs.items()}
        if not self.dictionary:
            obs, next_obs = obs['state'], next_obs['state']
        return (obs, torch.as_tensor(self.actions[index], device=device),
                torch.as_tensor(self.rewards[index], device=device), next_obs,
                torch.as_tensor(self.terminated[index], device=device))

    def state_dict(self) -> dict:
        """Compact, weights_only-safe payload, preserving ring order and sampling RNG."""
        return {
            'capacity': self.capacity, 'position': self.position, 'size': self.size,
            'dictionary': self.dictionary, 'rng': copy.deepcopy(self.rng.bit_generator.state),
            'obs': {key: torch.from_numpy(value[:self.size].copy()) for key, value in self.obs.items()},
            'next_obs': {key: torch.from_numpy(value[:self.size].copy()) for key, value in self.next_obs.items()},
            'actions': torch.from_numpy(self.actions[:self.size].copy()),
            'rewards': torch.from_numpy(self.rewards[:self.size].copy()),
            'terminated': torch.from_numpy(self.terminated[:self.size].copy()),
        }

    def load_state_dict(self, state: dict) -> None:
        size, position = int(state['size']), int(state['position'])
        if state['capacity'] != self.capacity or state['dictionary'] != self.dictionary:
            raise ValueError('Replay checkpoint capacity or observation type mismatch')
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError('Invalid replay cursor/size')
        if size < self.capacity and position != size:
            raise ValueError('Invalid partial replay cursor')
        # Validate the complete payload BEFORE mutating a live buffer.
        assignments = []
        for group in ('obs', 'next_obs'):
            target = getattr(self, group)
            if set(state[group]) != set(target):
                raise ValueError('Replay observation keys mismatch')
            for key, array in target.items():
                tensor = state[group][key]
                value = tensor.cpu().numpy()
                if value.shape != (size,) + array.shape[1:] or value.dtype != array.dtype:
                    raise ValueError('Replay observation shape/dtype mismatch')
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite replay observation')
                assignments.append((array, value))
        for name in ('actions', 'rewards', 'terminated'):
            array, value = getattr(self, name), state[name].cpu().numpy()
            if value.shape != (size,) + array.shape[1:] or value.dtype != array.dtype or not np.isfinite(value).all():
                raise ValueError(f'Invalid replay {name}')
            assignments.append((array, value))
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = copy.deepcopy(state['rng'])
        for array, value in assignments:
            array[:] = 0
            array[:size] = value
        self.position, self.size, self.rng = position, size, restored_rng


def global_rng_state() -> dict:
    state = np.random.get_state()
    result = {'torch': torch.get_rng_state(), 'python': random.getstate(),
              'numpy': [state[0], torch.from_numpy(state[1].copy()), state[2], state[3], state[4]]}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_global_rng(state: dict) -> None:
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], value[1].cpu().numpy(), value[2], value[3], value[4]))
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])



class TwinQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleList([ObservationEncoder(config.env.observation, config.env.neighbors, config.hidden_size, config.env.bev_frames) for _ in range(2)])
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size + 2, config.hidden_size), nn.ReLU(),
                                                 nn.Linear(config.hidden_size, 1)) for _ in range(2)])

    def forward(self, obs, actions):
        return tuple(head(torch.cat((encoder(obs), actions), dim=-1)).squeeze(-1)
                     for encoder, head in zip(self.encoders, self.heads))


class SACTrainer:
    def __init__(self, config, output, resume=None):
        if config.env.num_agents != 1:
            raise ValueError('SACTrainer supports a single learning ego')
        payload = None
        if resume:
            payload = torch.load(resume, map_location='cpu', weights_only=True)
            if payload.get('kind') != 'sac_training_state' or payload.get('format_version') != 1:
                raise ValueError('SAC resume requires training_state.pt, not an actor-only inference checkpoint')
            previous = copy.deepcopy(payload['config'])
            current = config.to_dict()
            # Resuming is not an implicit task/hyperparameter change.
            for key in ('total_steps', 'checkpoint_every', 'device', 'threads'):
                previous.pop(key, None)
                current.pop(key, None)
            if previous != current:
                raise ValueError('SAC resume config mismatch (observation, reward, replay and hyperparameters must match)')
            if config.total_steps <= payload['steps']:
                raise ValueError('Resume total_steps must exceed the saved absolute world-step counter')
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / 'training.csv').exists():
            raise FileExistsError('Choose a fresh output directory')
        set_seed(config.seed, config.threads)
        self.env = make_parallel_env(config.env)
        self.actor = model_from_config(config).to(config.device)
        self.critic = TwinQ(config).to(config.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.log_alpha = torch.tensor(math.log(0.2), device=config.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.initial = {key: param.detach().cpu().clone() for key, param in self.actor.named_parameters()}
        self.obs, _ = self.env.reset(seed=config.env.seed)
        self.buffer = ReplayBuffer(config.replay_size, self.obs['agent0'], config.seed, config.replay_max_bytes)
        self.rng = np.random.default_rng(config.seed)
        save_config(config, self.output / 'config.yaml')
        save_json(runtime_manifest(), self.output / 'runtime.json')
        self.writer = ScalarWriter(self.output / 'tensorboard')
        self.steps, self.updates, self.episodes = 0, 0, 0
        self.resumed = bool(payload)
        if payload:
            self._restore(payload)
            self.initial = {key: param.detach().cpu().clone() for key, param in self.actor.named_parameters()}

    def _restore(self, payload: dict) -> None:
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.target.load_state_dict(payload['target'])
        self.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
        with torch.no_grad():
            self.log_alpha.copy_(payload['log_alpha'].to(self.config.device))
        self.alpha_optimizer.load_state_dict(payload['alpha_optimizer'])
        self.buffer.load_state_dict(payload['replay'])
        self.rng.bit_generator.state = copy.deepcopy(payload['exploration_rng'])
        self.steps, self.updates, self.episodes = payload['steps'], payload['updates'], payload['episodes']
        # Only a new episode is reconstructed; no physical state is fabricated.
        if payload.get('environment_rng') is not None and hasattr(self.env, 'rng'):
            self.env.rng.bit_generator.state = copy.deepcopy(payload['environment_rng'])
            self.obs, _ = self.env.reset()
        restore_global_rng(payload['global_rng'])

    def save_training_checkpoint(self, path) -> None:
        '''Atomically save complete SAC learning state; physics restarts on resume.'''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'kind': 'sac_training_state', 'format_version': 1, 'config': self.config.to_dict(),
            'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
            'target': self.target.state_dict(), 'log_alpha': self.log_alpha.detach(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(), 'replay': self.buffer.state_dict(),
            'steps': self.steps, 'updates': self.updates, 'episodes': self.episodes,
            'global_rng': global_rng_state(), 'exploration_rng': copy.deepcopy(self.rng.bit_generator.state),
            'environment_rng': copy.deepcopy(self.env.rng.bit_generator.state) if hasattr(self.env, 'rng') else None,
            'resume_semantics': 'complete_learning_state_new_episode_not_physics_continuation',
            'checkpoint_was_mid_episode': bool(getattr(self.env, 'steps', 0)),
        }
        temporary = path.with_suffix(path.suffix + '.tmp')
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


    def update(self):
        cfg = self.config
        obs, action, reward, next_obs, terminated = self.buffer.sample(cfg.batch_size, cfg.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, _, next_logp, _ = self.actor.sample(next_obs)
            q1_target, q2_target = self.target(next_obs, next_action)
            target = reward + cfg.gamma * (1 - terminated) * (torch.minimum(q1_target, q2_target) - alpha * next_logp)
        q1, q2 = self.critic(obs, action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        policy_action, _, logp, _ = self.actor.sample(obs)
        q1, q2 = self.critic(obs, policy_action)
        actor_loss = (alpha * logp - torch.minimum(q1, q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (logp.detach() - 2.0)).mean()  # target entropy = -2
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
                target_param.lerp_(param, cfg.tau)
        if not all(torch.isfinite(value).all() for value in (actor_loss, critic_loss, alpha_loss)):
            raise FloatingPointError('Non-finite SAC loss')
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(),
                'alpha': self.log_alpha.exp().item(), 'alpha_loss': alpha_loss.item()}

    def train(self):
        cfg = self.config
        start = time.perf_counter()
        start_steps = self.steps
        start_updates = self.updates
        start_episodes = self.episodes
        metrics = {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha': 0.2, 'alpha_loss': 0.0}
        try:
            for step in range(self.steps + 1, cfg.total_steps + 1):
                self.steps = step
                if step <= cfg.learning_starts:
                    action = self.rng.uniform(-1, 1, 2).astype(np.float32)
                else:
                    action = self.actor.predict(self.obs['agent0'], deterministic=False)
                next_obs, rewards, terms, truncs, infos = self.env.step({'agent0': action})
                self.buffer.add(self.obs['agent0'], action, rewards['agent0'], next_obs['agent0'], terms['agent0'])
                if step >= cfg.learning_starts and self.buffer.size >= cfg.batch_size:
                    metrics = self.update()
                    self.updates += 1
                if terms['agent0'] or truncs['agent0']:
                    self.episodes += 1
                    info = infos['agent0']
                    append_csv(self.output / 'episodes.csv', {'world_step': step, 'return': info.get('episode_return', 0),
                        'success': info['success'], 'collision': info['collision'], 'out_of_road': info['out_of_road'],
                        'truncated': truncs['agent0'], 'completion': info['route_completion']})
                    self.obs, _ = self.env.reset()
                else:
                    self.obs = next_obs
                if step % cfg.rollout_steps == 0 or step == cfg.total_steps:
                    row = {'world_steps': step, 'updates': self.updates, 'elapsed_seconds': time.perf_counter() - start,
                           'world_steps_per_second': (step - start_steps) / (time.perf_counter() - start), **metrics}
                    append_csv(self.output / 'training.csv', row)
                    for key, value in metrics.items():
                        self.writer.add_scalar('sac/' + key, value, step)
                if step % cfg.checkpoint_every == 0:
                    save_checkpoint(self.output / 'checkpoints' / f'step_{step}.pt', self.actor, cfg, steps=step,
                                    extra={'purpose': 'SAC inference; full resume uses the paired .resume.pt'})
                    self.save_training_checkpoint(self.output / 'checkpoints' / f'step_{step}.resume.pt')
            path = self.output / 'model.pt'
            save_checkpoint(path, self.actor, cfg, steps=cfg.total_steps,
                extra={'purpose': 'SAC inference; full learning state is in training_state.pt'})
            self.save_training_checkpoint(self.output / 'training_state.pt')
            changes = {key: float((param.detach().cpu() - self.initial[key]).abs().sum()) for key, param in self.actor.named_parameters()}
            result = {'backend': cfg.env.backend, 'algorithm': 'sac', 'observation': cfg.env.observation,
                'num_agents': 1, 'world_steps_total': cfg.total_steps, 'updates': self.updates,
                'elapsed_seconds': time.perf_counter() - start, 'episodes_finished': self.episodes,
                'world_steps_this_run': self.steps - start_steps, 'updates_this_run': self.updates - start_updates,
                'episodes_finished_this_run': self.episodes - start_episodes, 'resumed': self.resumed,
                'resume_semantics': 'complete_learning_state_new_episode_not_physics_continuation',
                'training_state_sha256': sha256(self.output / 'training_state.pt'),
                'model_sha256': sha256(path), 'parameters_changed_l1': sum(changes.values()),
                'encoder_changed_l1': sum(v for k, v in changes.items() if k.startswith('encoder.')),
                'status': 'training_completed_not_a_driving_safety_claim'}
            save_json(result, self.output / 'result.json')
            return result
        finally:
            self.env.close()
            self.writer.close()


class SACAgent:
    def __init__(self, checkpoint, device='cpu'):
        self.policy, self.config, self.metadata = load_checkpoint(checkpoint, device)

    def predict(self, observation, deterministic=True):
        return self.policy.predict(observation, deterministic)

    def reset(self):
        pass
