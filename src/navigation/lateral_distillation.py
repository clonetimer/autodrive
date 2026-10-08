"""Closed-form lateral-head distillation from policy-visible route targets.

The route teacher is deterministic and uses only the same state vector exposed
to the actor.  To avoid disturbing learned longitudinal behavior, the encoder
and longitudinal actor row remain bit-for-bit unchanged.  The steering row is
fit in latent (atanh action) space with ridge regression, matching the policy's
``tanh(linear(features))`` parameterization exactly.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch


@dataclass(frozen=True)
class LateralHeadFitResult:
    samples: int
    feature_dim: int
    ridge: float
    initial_action_mse: float
    final_action_mse: float
    initial_peak_abs_error: float
    final_peak_abs_error: float
    encoder_changed_l1: float
    longitudinal_actor_changed_l1: float
    steering_actor_changed_l1: float
    condition_number: float

    def to_dict(self):
        return asdict(self)


def fit_lateral_head_ridge(model, states, target_steering, *, ridge: float = 1e-3,
                           device: str = 'cpu') -> LateralHeadFitResult:
    states = np.asarray(states, dtype=np.float32)
    targets = np.asarray(target_steering, dtype=np.float32).reshape(-1)
    if states.ndim != 2 or len(states) == 0 or targets.shape != (len(states),):
        raise ValueError('Lateral-head fit expects states[N,D] and steering[N]')
    if not np.isfinite(states).all() or not np.isfinite(targets).all() or np.any(np.abs(targets) > 1):
        raise ValueError('Lateral-head fit arrays must be finite normalized controls')
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError('ridge must be finite and nonnegative')

    model = model.to(device).eval()
    before_encoder = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('encoder.')}
    before_long_w = model.actor.weight[1].detach().cpu().clone()
    before_long_b = model.actor.bias[1].detach().cpu().clone()
    before_steer_w = model.actor.weight[0].detach().cpu().clone()
    before_steer_b = model.actor.bias[0].detach().cpu().clone()
    with torch.no_grad():
        obs = torch.as_tensor(states, dtype=torch.float32, device=device)
        features = model.encoder(obs).detach().cpu().numpy().astype(np.float64)
        initial = model.predict  # keep attribute resolution out of loop below
        pred0 = np.tanh(features @ before_steer_w.numpy().astype(np.float64) + float(before_steer_b))

    # Tanh inverse maps desired normalized steering to the actor's latent mean.
    clipped = np.clip(targets.astype(np.float64), -0.999, 0.999)
    latent_target = np.arctanh(clipped)
    x = np.concatenate((features, np.ones((len(features), 1), dtype=np.float64)), axis=1)
    gram = x.T @ x
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0  # do not regularize bias
    condition = float(np.linalg.cond(gram + penalty))
    beta = np.linalg.solve(gram + penalty, x.T @ latent_target)
    if not np.isfinite(beta).all():
        raise FloatingPointError('Non-finite lateral ridge solution')
    with torch.no_grad():
        model.actor.weight[0].copy_(torch.as_tensor(beta[:-1], dtype=model.actor.weight.dtype, device=device))
        model.actor.bias[0].copy_(torch.as_tensor(beta[-1], dtype=model.actor.bias.dtype, device=device))
    pred1 = np.tanh(x @ beta)

    encoder_change = sum(float((p.detach().cpu()-before_encoder[n]).abs().sum())
                         for n,p in model.named_parameters() if n in before_encoder)
    long_change = float((model.actor.weight[1].detach().cpu()-before_long_w).abs().sum()
                        +(model.actor.bias[1].detach().cpu()-before_long_b).abs().sum())
    steer_change = float((model.actor.weight[0].detach().cpu()-before_steer_w).abs().sum()
                         +(model.actor.bias[0].detach().cpu()-before_steer_b).abs().sum())
    return LateralHeadFitResult(
        samples=len(states), feature_dim=features.shape[1], ridge=float(ridge),
        initial_action_mse=float(np.mean((pred0-targets)**2)),
        final_action_mse=float(np.mean((pred1-targets)**2)),
        initial_peak_abs_error=float(np.max(np.abs(pred0-targets))),
        final_peak_abs_error=float(np.max(np.abs(pred1-targets))),
        encoder_changed_l1=encoder_change,
        longitudinal_actor_changed_l1=long_change,
        steering_actor_changed_l1=steer_change,
        condition_number=condition,
    )
