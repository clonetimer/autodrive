"""Intervention-aware policy distillation for the transparent safety supervisor.

The supervisor is a heuristic longitudinal teacher, not a safety certificate.
Targets are the actions actually applied while the current actor controls the
same simulator state.  v2.6 additionally supports a *longitudinal-only* update
that freezes the encoder and the steering actor row.  This is useful when the
teacher never changes steering: safety corrections can be absorbed without
silently degrading an already learned lateral policy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from src.agents.distillation import actor_actions


@dataclass(frozen=True)
class SafetyDistillationResult:
    samples: int
    intervention_samples: int
    epochs: int
    batches: int
    initial_mse: float
    final_mse: float
    initial_intervention_mse: float | None
    final_intervention_mse: float | None
    initial_nonintervention_mse: float | None
    final_nonintervention_mse: float | None
    actor_encoder_changed_l1: float
    encoder_changed_l1: float
    steering_actor_changed_l1: float
    longitudinal_actor_changed_l1: float
    intervention_weight: float
    steering_weight: float
    train_encoder: bool
    longitudinal_only: bool
    steering_only: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_arrays(states, targets, intervened, severity):
    states = np.asarray(states, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    intervened = np.asarray(intervened, dtype=bool)
    severity = np.asarray(severity, dtype=np.float32)
    n = len(targets)
    if states.ndim != 2 or n == 0 or len(states) != n:
        raise ValueError('Safety distillation states must be a non-empty [N,D] array')
    if targets.shape != (n, 2) or not np.isfinite(targets).all():
        raise ValueError('Safety distillation targets must be finite [N,2] actions')
    if intervened.shape != (n,) or severity.shape != (n,) or not np.isfinite(severity).all():
        raise ValueError('Intervention/severity arrays must align with samples')
    if np.any((severity < 0) | (severity > 1)):
        raise ValueError('Safety severity must lie in [0,1]')
    if not np.isfinite(states).all():
        raise ValueError('Safety distillation states must be finite')
    return states, targets, intervened, severity


def _weighted_loss(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
                   steering_weight: float) -> torch.Tensor:
    component = torch.as_tensor([steering_weight, 1.0], dtype=predicted.dtype, device=predicted.device)
    per_sample = ((predicted - target).square() * component).mean(dim=-1)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def _plain_mse(model, states: np.ndarray, targets: np.ndarray, mask: np.ndarray | None,
               device: str) -> float | None:
    if mask is not None:
        if not np.any(mask):
            return None
        states, targets = states[mask], targets[mask]
    with torch.no_grad():
        obs = torch.as_tensor(states, dtype=torch.float32, device=device)
        target = torch.as_tensor(targets, dtype=torch.float32, device=device)
        pred = actor_actions(model, obs)
        return float(nn.functional.mse_loss(pred, target).item())


def _change_breakdown(model, before: dict[str, torch.Tensor]) -> tuple[float, float, float, float]:
    encoder, steering, longitudinal = 0.0, 0.0, 0.0
    for name, value in model.named_parameters():
        if name not in before:
            continue
        delta = (value.detach().cpu() - before[name]).abs()
        if name.startswith('encoder.'):
            encoder += float(delta.sum())
        elif name == 'actor.weight':
            steering += float(delta[0].sum())
            longitudinal += float(delta[1].sum())
        elif name == 'actor.bias':
            steering += float(delta[0])
            longitudinal += float(delta[1])
    return encoder + steering + longitudinal, encoder, steering, longitudinal


def distill_safety_actor(model, states, targets, intervened, severity, *, epochs: int = 20,
                         batch_size: int = 256, learning_rate: float = 2e-4,
                         intervention_weight: float = 4.0, steering_weight: float = 0.35,
                         train_encoder: bool = True, longitudinal_only: bool = False,
                         steering_only: bool = False,
                         device: str = 'cpu', seed: int = 0) -> SafetyDistillationResult:
    """Fit the deployment actor to supervisor-applied actions.

    ``train_encoder=False, longitudinal_only=True`` is the strict longitudinal
    preservation mode introduced in v2.6.  v2.7 adds the symmetric
    ``steering_only=True`` mode for privileged path-teacher imitation: critic,
    log-std, encoder and the longitudinal actor row remain byte-for-byte
    unchanged while only the steering actor row is fitted. Legacy encoder+actor
    training remains the default for backwards-compatible reproduction.
    """
    states, targets, intervened, severity = _validate_arrays(states, targets, intervened, severity)
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError('Invalid safety-distillation optimizer settings')
    if intervention_weight < 0 or not (steering_weight > 0):
        raise ValueError('Invalid safety-distillation weights')
    if longitudinal_only and steering_only:
        raise ValueError('longitudinal_only and steering_only are mutually exclusive')
    if (longitudinal_only or steering_only) and train_encoder:
        # Updating shared features can still alter the supposedly preserved action
        # component even when its linear row receives zero gradient. Strict
        # component-only modes therefore require a frozen encoder.
        name = 'longitudinal_only' if longitudinal_only else 'steering_only'
        raise ValueError(f'{name} requires train_encoder=False to preserve the other action exactly')

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    model.train()
    before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
              if name.startswith('encoder.') or name.startswith('actor.')}
    encoder_parameters = list(model.encoder.parameters())
    if not train_encoder:
        for parameter in encoder_parameters:
            parameter.requires_grad_(False)
    parameters = list(model.actor.parameters())
    if train_encoder:
        parameters = encoder_parameters + parameters
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    initial = _plain_mse(model, states, targets, None, device)
    initial_i = _plain_mse(model, states, targets, intervened, device)
    initial_n = _plain_mse(model, states, targets, ~intervened, device)
    batches = 0
    n = len(targets)
    if not np.any(intervened):
        model.eval()
        if not train_encoder:
            for parameter in encoder_parameters:
                parameter.requires_grad_(True)
        total, enc, steer, longitudinal = _change_breakdown(model, before)
        return SafetyDistillationResult(
            samples=n, intervention_samples=0, epochs=0, batches=0,
            initial_mse=float(initial), final_mse=float(initial),
            initial_intervention_mse=None, final_intervention_mse=None,
            initial_nonintervention_mse=initial_n, final_nonintervention_mse=initial_n,
            actor_encoder_changed_l1=total, encoder_changed_l1=enc,
            steering_actor_changed_l1=steer, longitudinal_actor_changed_l1=longitudinal,
            intervention_weight=float(intervention_weight), steering_weight=float(steering_weight),
            train_encoder=bool(train_encoder), longitudinal_only=bool(longitudinal_only),
            steering_only=bool(steering_only))

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            obs = torch.as_tensor(states[idx], dtype=torch.float32, device=device)
            target = torch.as_tensor(targets[idx], dtype=torch.float32, device=device)
            severity_batch = torch.as_tensor(severity[idx], dtype=torch.float32, device=device)
            intervention_batch = torch.as_tensor(intervened[idx].astype(np.float32), device=device)
            weights = 1.0 + intervention_weight * intervention_batch * (0.5 + 0.5 * severity_batch)
            predicted = actor_actions(model, obs)
            loss = _weighted_loss(predicted, target, weights, steering_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite safety-distillation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if longitudinal_only:
                # Linear actor is [steering, longitudinal]. Adam has no weight
                # decay here, so a zero gradient keeps the steering row exact.
                if model.actor.weight.grad is not None:
                    model.actor.weight.grad[0].zero_()
                if model.actor.bias.grad is not None:
                    model.actor.bias.grad[0].zero_()
            elif steering_only:
                # Symmetric strict lateral-imitation mode: preserve the learned
                # longitudinal actor row byte-for-byte while fitting steering.
                if model.actor.weight.grad is not None:
                    model.actor.weight.grad[1].zero_()
                if model.actor.bias.grad is not None:
                    model.actor.bias.grad[1].zero_()
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            batches += 1

    model.eval()
    if not train_encoder:
        for parameter in encoder_parameters:
            parameter.requires_grad_(True)
    final = _plain_mse(model, states, targets, None, device)
    final_i = _plain_mse(model, states, targets, intervened, device)
    final_n = _plain_mse(model, states, targets, ~intervened, device)
    total, enc, steer, longitudinal = _change_breakdown(model, before)
    return SafetyDistillationResult(
        samples=n, intervention_samples=int(intervened.sum()), epochs=epochs, batches=batches,
        initial_mse=float(initial), final_mse=float(final),
        initial_intervention_mse=initial_i, final_intervention_mse=final_i,
        initial_nonintervention_mse=initial_n, final_nonintervention_mse=final_n,
        actor_encoder_changed_l1=total, encoder_changed_l1=enc,
        steering_actor_changed_l1=steer, longitudinal_actor_changed_l1=longitudinal,
        intervention_weight=float(intervention_weight), steering_weight=float(steering_weight),
        train_encoder=bool(train_encoder), longitudinal_only=bool(longitudinal_only),
        steering_only=bool(steering_only))
