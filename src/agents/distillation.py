"""Offline actor distillation utilities for cross-observation driving policies.

Distillation is deliberately separated from RL training: a privileged teacher
may provide actions while the student is trained only from its deployment
observation.  The saved evidence labels this provenance explicitly so a
behavior-cloned RGB policy is never reported as an independently learned RL
policy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from src.perception.encoder import tensor_observation


@dataclass(frozen=True)
class DistillationResult:
    samples: int
    epochs: int
    batches: int
    initial_mse: float
    final_mse: float
    actor_encoder_changed_l1: float

    def to_dict(self) -> dict:
        return asdict(self)


def _slice_observation(observation: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in observation.items()}


def actor_actions(model, observation) -> torch.Tensor:
    distribution, _ = model.distribution(observation)
    return torch.tanh(distribution.mean)


def distill_actor(model, observations: dict[str, np.ndarray], targets: np.ndarray, *,
                  epochs: int = 20, batch_size: int = 64, learning_rate: float = 3e-4,
                  device: str = 'cpu', seed: int = 0) -> DistillationResult:
    """Fit encoder+actor to deterministic teacher actions using MSE.

    Critic and exploration log-std are intentionally untouched.  This makes the
    artifact suitable as a deployment actor or as a warm start for subsequent
    PPO, while avoiding fabricated value-function semantics.
    """
    if not observations or any(len(v) != len(targets) for v in observations.values()):
        raise ValueError('Observation arrays and targets must have one common non-empty length')
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError('Teacher targets must be finite [N,2] actions')
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError('Invalid distillation optimizer settings')
    n = len(targets)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    model.train()
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
              if name.startswith('encoder.') or name.startswith('actor.')}
    parameters = list(model.encoder.parameters()) + list(model.actor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    loss_fn = nn.MSELoss()

    all_obs = tensor_observation(observations, device)
    all_targets = torch.as_tensor(targets, device=device)
    with torch.no_grad():
        initial = float(loss_fn(actor_actions(model, all_obs), all_targets).item())
    batches = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if not len(idx):
                continue
            batch_obs = tensor_observation(_slice_observation(observations, idx), device)
            batch_target = torch.as_tensor(targets[idx], device=device)
            loss = loss_fn(actor_actions(model, batch_obs), batch_target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite distillation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            batches += 1
    model.eval()
    with torch.no_grad():
        final = float(loss_fn(actor_actions(model, all_obs), all_targets).item())
    changed = 0.0
    for name, value in model.named_parameters():
        if name in before:
            changed += float((value.detach().cpu() - before[name]).abs().sum())
    return DistillationResult(n, epochs, batches, initial, final, changed)
