"""Anchored privileged-expert distillation for MetaDrive state policies.

The objective learns expert steering/safety targets while explicitly anchoring
the student to the frozen source actor and source representation.  This is used
after v2.7 experiments showed unconstrained encoder+actor imitation can reduce
teacher MSE yet catastrophically drift off the source policy distribution.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from src.agents.distillation import actor_actions


@dataclass(frozen=True)
class AnchoredExpertDistillationResult:
    samples: int
    intervention_samples: int
    epochs: int
    batches: int
    initial_teacher_mse: float
    final_teacher_mse: float
    initial_reference_mse: float
    final_reference_mse: float
    encoder_changed_l1: float
    steering_actor_changed_l1: float
    longitudinal_actor_changed_l1: float
    reference_action_coef: float
    feature_anchor_coef: float
    intervention_weight: float

    def to_dict(self) -> dict:
        return asdict(self)


def _changes(model, before):
    enc = steer = longitudinal = 0.0
    for name, param in model.named_parameters():
        if name not in before:
            continue
        delta = (param.detach().cpu() - before[name]).abs()
        if name.startswith('encoder.'):
            enc += float(delta.sum())
        elif name == 'actor.weight':
            steer += float(delta[0].sum()); longitudinal += float(delta[1].sum())
        elif name == 'actor.bias':
            steer += float(delta[0]); longitudinal += float(delta[1])
    return enc, steer, longitudinal


def distill_anchored_expert(model, reference_model, states, targets, intervened, severity, *,
                            epochs: int = 20, batch_size: int = 256,
                            learning_rate: float = 1e-4, intervention_weight: float = 3.0,
                            longitudinal_teacher_weight: float = .25,
                            reference_action_coef: float = 1.0,
                            feature_anchor_coef: float = .05,
                            device: str = 'cpu', seed: int = 0) -> AnchoredExpertDistillationResult:
    states = np.asarray(states, np.float32); targets = np.asarray(targets, np.float32)
    intervened = np.asarray(intervened, bool); severity = np.asarray(severity, np.float32)
    n = len(states)
    if states.ndim != 2 or targets.shape != (n, 2) or intervened.shape != (n,) or severity.shape != (n,) or n == 0:
        raise ValueError('Anchored expert arrays must align as state[N,D], target[N,2], masks[N]')
    if not np.isfinite(states).all() or not np.isfinite(targets).all() or not np.isfinite(severity).all():
        raise ValueError('Anchored expert arrays must be finite')
    if np.any((severity < 0) | (severity > 1)):
        raise ValueError('severity must lie in [0,1]')
    if min(epochs, batch_size) <= 0 or learning_rate <= 0 or intervention_weight < 0:
        raise ValueError('Invalid anchored-distillation optimizer settings')
    if not 0 <= longitudinal_teacher_weight <= 1 or reference_action_coef < 0 or feature_anchor_coef < 0:
        raise ValueError('Invalid anchored-distillation loss weights')

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = model.to(device); reference_model = reference_model.to(device)
    reference_model.eval()
    for p in reference_model.parameters(): p.requires_grad_(False)
    before = {name:p.detach().cpu().clone() for name,p in model.named_parameters()
              if name.startswith('encoder.') or name.startswith('actor.')}
    parameters = list(model.encoder.parameters()) + list(model.actor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    component = torch.tensor([1.0, longitudinal_teacher_weight], dtype=torch.float32, device=device)
    preserve_component = torch.tensor([.5, 1.0], dtype=torch.float32, device=device)

    def metrics():
        model.eval(); teacher_sum = ref_sum = count = 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                obs = torch.as_tensor(states[start:start+batch_size], device=device)
                tgt = torch.as_tensor(targets[start:start+batch_size], device=device)
                pred = actor_actions(model, obs); ref = actor_actions(reference_model, obs)
                teacher_sum += float(nn.functional.mse_loss(pred, tgt, reduction='sum'))
                ref_sum += float(nn.functional.mse_loss(pred, ref, reduction='sum'))
                count += pred.numel()
        return teacher_sum / count, ref_sum / count

    initial_teacher, initial_reference = metrics(); batches = 0
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start+batch_size]
            if not len(idx): continue
            obs = torch.as_tensor(states[idx], device=device)
            tgt = torch.as_tensor(targets[idx], device=device)
            mask = torch.as_tensor(intervened[idx].astype(np.float32), device=device)
            sev = torch.as_tensor(severity[idx], device=device)
            with torch.no_grad():
                ref_action = actor_actions(reference_model, obs)
                ref_features = reference_model.encoder(obs)
            features = model.encoder(obs)
            predicted = torch.tanh(model.actor(features))
            weights = 1.0 + intervention_weight * mask * (.5 + .5 * sev)
            per = ((predicted - tgt).square() * component).mean(-1)
            teacher_loss = (per * weights).sum() / weights.sum().clamp_min(1e-8)
            reference_loss = ((predicted - ref_action).square() * preserve_component).mean()
            feature_loss = nn.functional.mse_loss(features, ref_features)
            loss = teacher_loss + reference_action_coef * reference_loss + feature_anchor_coef * feature_loss
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite anchored expert loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step(); batches += 1
    final_teacher, final_reference = metrics(); model.eval()
    enc, steer, longitudinal = _changes(model, before)
    return AnchoredExpertDistillationResult(n, int(intervened.sum()), epochs, batches,
        float(initial_teacher), float(final_teacher), float(initial_reference), float(final_reference),
        enc, steer, longitudinal, float(reference_action_coef), float(feature_anchor_coef), float(intervention_weight))
