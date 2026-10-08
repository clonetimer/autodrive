"""Learned lateral residual adapter distilled from transparent RouteAssist.

The adapter is intentionally separate from the base PPO checkpoint.  It receives
only the policy-visible state vector and the base actor's normalized action.  At
runtime it predicts (1) a bounded steering residual and (2) how strongly that
residual should be applied.  No pure-pursuit geometry is evaluated at inference.

This lets the project measure whether rule assistance can be transferred into a
learned component instead of hiding the rule controller behind an "RL" label.
The artifact stores the base-model SHA-256 and refuses to attach to a different
actor unless verification is explicitly disabled.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Any
import hashlib
import json
import numpy as np
import torch
from torch import nn

from src.end_to_end.agent import EndToEndAgent
from src.utils.logger import sha256


@dataclass(frozen=True)
class RouteAdapterConfig:
    state_dim: int
    hidden_size: int = 96
    max_blend: float = 0.5
    max_raw_correction: float = 2.0

    def __post_init__(self):
        if self.state_dim <= 0 or self.hidden_size <= 0:
            raise ValueError('Route adapter dimensions must be positive')
        if not 0 < self.max_blend <= 1:
            raise ValueError('max_blend must lie in (0,1]')
        if not 0 < self.max_raw_correction <= 2:
            raise ValueError('max_raw_correction must lie in (0,2]')




def policy_visible_curve_score(state: torch.Tensor) -> torch.Tensor:
    """Approximate route-turn severity from policy-visible route preview only.

    The four preview points occupy state[12:20] and are normalized by [60m,20m].
    This helper is used for training emphasis/diagnostics, not as a runtime
    steering controller.
    """
    if state.ndim != 2 or state.shape[-1] < 20:
        raise ValueError('curve score expects batched policy state with route preview')
    points = state[:, 12:20].reshape(-1, 4, 2)
    x = points[..., 0] * 60.0
    y = points[..., 1] * 20.0
    angles = torch.atan2(y, x.clamp(min=1e-3))
    severity = angles.abs().amax(dim=1) / 0.55
    return severity.clamp(0.0, 1.0)

class LearnedRouteAdapter(nn.Module):
    """Predict a learned approximation of adaptive route-assist correction."""

    def __init__(self, config: RouteAdapterConfig):
        super().__init__()
        self.config = config
        in_dim = config.state_dim + 2
        h = config.hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h // 2), nn.SiLU(),
        )
        self.delta_head = nn.Linear(h // 2, 1)
        self.blend_head = nn.Linear(h // 2, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Start near identity: no residual and near-zero learned blend.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.blend_head.weight)
        nn.init.constant_(self.blend_head.bias, -4.0)

    def forward(self, state: torch.Tensor, base_action: torch.Tensor):
        if state.ndim != 2 or base_action.ndim != 2 or base_action.shape[-1] != 2:
            raise ValueError('Adapter expects batched state[N,D] and base_action[N,2]')
        if state.shape[-1] != self.config.state_dim or state.shape[0] != base_action.shape[0]:
            raise ValueError('Adapter input shape mismatch')
        features = self.backbone(torch.cat((state.float(), base_action.float()), dim=-1))
        raw_delta = torch.tanh(self.delta_head(features)).squeeze(-1) * self.config.max_raw_correction
        blend = torch.sigmoid(self.blend_head(features)).squeeze(-1) * self.config.max_blend
        correction = raw_delta * blend
        steering = (base_action[:, 0] + correction).clamp(-1.0, 1.0)
        return steering, raw_delta, blend

    @torch.no_grad()
    def adapt(self, state, base_action, device: str | torch.device = 'cpu') -> tuple[np.ndarray, dict[str, float]]:
        state_np = np.asarray(state, dtype=np.float32)
        base_np = np.asarray(base_action, dtype=np.float32)
        if state_np.shape != (self.config.state_dim,) or base_np.shape != (2,):
            raise ValueError('Adapter expects one state vector and [steering,longitudinal] action')
        if not np.isfinite(state_np).all() or not np.isfinite(base_np).all():
            raise ValueError('Adapter inputs must be finite')
        state_t = torch.as_tensor(state_np[None], dtype=torch.float32, device=device)
        action_t = torch.as_tensor(base_np[None], dtype=torch.float32, device=device)
        steering, raw_delta, blend = self(state_t, action_t)
        adapted = base_np.copy()
        adapted[0] = float(steering[0].cpu())
        diagnostics = {
            'learned_adapter_raw_correction': float(raw_delta[0].cpu()),
            'learned_adapter_blend': float(blend[0].cpu()),
            'learned_adapter_steering_correction': abs(float(adapted[0] - base_np[0])),
            'learned_adapter_base_steering': float(base_np[0]),
            'learned_adapter_applied_steering': float(adapted[0]),
        }
        return adapted, diagnostics


def _dataset_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode('utf-8'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def save_route_adapter(path: str | Path, model: LearnedRouteAdapter, *, base_model: str | Path,
                       training: dict[str, Any], dataset: dict[str, np.ndarray] | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'learned_route_adapter_v1',
        'config': asdict(model.config),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'base_model_path': str(base_model),
        'base_model_sha256': sha256(base_model),
        'training': training,
        'dataset_sha256': _dataset_fingerprint(dataset) if dataset is not None else None,
    }
    torch.save(payload, path)
    return {'path': str(path), 'sha256': sha256(path), 'base_model_sha256': payload['base_model_sha256'],
            'dataset_sha256': payload['dataset_sha256']}


def load_route_adapter(path: str | Path, *, base_model: str | Path | None = None,
                       device: str = 'cpu', verify_base: bool = True):
    payload = torch.load(path, map_location=device, weights_only=False)
    format_name = payload.get('format')
    if format_name == 'learned_route_adapter_v1':
        config = RouteAdapterConfig(**payload['config'])
        model = LearnedRouteAdapter(config).to(device)
    elif format_name == 'curve_route_adapter_v2':
        config = CurveRouteAdapterConfig(**payload['config'])
        model = CurveRouteAdapter(config).to(device)
    elif format_name == 'scene_aware_temporal_adapter_v3':
        # Local import avoids a module cycle: the v3 adapter reuses the common
        # agent/artifact loader but its scene encoder lives in a separate module.
        from src.navigation.scene_aware_adapter import (
            SceneAwareTemporalAdapter, SceneAwareTemporalAdapterConfig)
        config = SceneAwareTemporalAdapterConfig(**payload['config'])
        model = SceneAwareTemporalAdapter(config).to(device)
    elif format_name == 'joint_action_temporal_adapter_v4':
        from src.navigation.joint_action_adapter import (
            JointActionTemporalAdapter, JointActionAdapterConfig)
        config = JointActionAdapterConfig(**payload['config'])
        model = JointActionTemporalAdapter(config).to(device)
    else:
        raise ValueError('Unsupported route-adapter artifact format')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    if base_model is not None and verify_base:
        actual = sha256(base_model)
        if actual != payload.get('base_model_sha256'):
            raise ValueError('Route adapter was trained for a different base policy SHA-256')
    return model, payload


class LearnedRouteAdapterAgent:
    """Compose a frozen PPO actor with a learned steering adapter.

    This agent performs no geometric controller call at inference.  Longitudinal
    SafetySupervisor, when enabled, remains an environment-level system component
    and must still be reported separately.
    """

    def __init__(self, base_checkpoint: str | Path, adapter_checkpoint: str | Path, device: str = 'cpu'):
        self.base = EndToEndAgent(base_checkpoint, device=device)
        self.policy = self.base.policy  # evaluator neural warm-up compatibility
        self.adapter, self.adapter_payload = load_route_adapter(
            adapter_checkpoint, base_model=base_checkpoint, device=device, verify_base=True)
        self.device = device
        self.base_fingerprint = self.base.fingerprint
        self.adapter_fingerprint = sha256(adapter_checkpoint)
        self.fingerprint = hashlib.sha256(
            (self.base_fingerprint + ':' + self.adapter_fingerprint).encode('ascii')).hexdigest()
        self.last_diagnostics: dict[str, float] = {}
        self._requires_history = bool(getattr(self.adapter, 'requires_history', False))
        history_len = int(getattr(self.adapter.config, 'history_len', 1)) if self._requires_history else 1
        self._state_history = deque(maxlen=history_len)

    def reset(self) -> None:
        self.last_diagnostics = {}
        self._state_history.clear()
        self.base.reset()

    def predict(self, observation, deterministic: bool = True) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float32)
        base_action = self.base.predict(state, deterministic=deterministic)
        if self._requires_history:
            self._state_history.append(state.copy())
            history = np.stack(tuple(self._state_history), axis=0)
            adapted, diagnostics = self.adapter.adapt(history, base_action, self.device)
        else:
            adapted, diagnostics = self.adapter.adapt(state, base_action, self.device)
        self.last_diagnostics = diagnostics
        return adapted

    def act(self, observations: dict[str, object]) -> dict[str, np.ndarray]:
        return {name: self.predict(obs) for name, obs in observations.items()}


@dataclass(frozen=True)
class CurveRouteAdapterConfig:
    """Direct policy-visible lateral residual used by v3.2 curve specialization."""
    state_dim: int
    hidden_size: int = 96
    max_correction: float = 0.18

    def __post_init__(self):
        if self.state_dim < 20 or self.hidden_size <= 0:
            raise ValueError('Curve adapter requires state_dim>=20 and positive hidden size')
        if not 0 < self.max_correction <= 0.5:
            raise ValueError('max_correction must lie in (0,0.5]')


def _route_geometry_tensor(state: torch.Tensor) -> torch.Tensor:
    """Policy-visible route features; deterministic feature extraction, not control."""
    route = state[:, 12:20].reshape(-1, 4, 2)
    x = route[..., 0] * 60.0
    y = route[..., 1] * 20.0
    angles = torch.atan2(y, x.clamp(min=1e-3))
    # First differences expose how rapidly route heading changes with lookahead.
    delta = torch.zeros_like(angles)
    delta[:, 0] = angles[:, 0]
    delta[:, 1:] = angles[:, 1:] - angles[:, :-1]
    return torch.cat((angles / 0.7, delta / 0.35), dim=-1).clamp(-5, 5)


class CurveRouteAdapter(nn.Module):
    """Direct residual steering model with a dedicated route-intent encoder.

    Unlike the v3.1 adapter, this model does not decompose the target into a
    raw pure-pursuit delta and a blend. It learns the *applied* teacher
    correction directly. This removes an unnecessary factorization that was
    poorly identified under fixed-blend curve supervision.
    """
    def __init__(self, config: CurveRouteAdapterConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.ego_encoder = nn.Sequential(
            nn.Linear(12, h // 2), nn.LayerNorm(h // 2), nn.SiLU(),
            nn.Linear(h // 2, h // 2), nn.SiLU(),
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(16, h // 2), nn.LayerNorm(h // 2), nn.SiLU(),
            nn.Linear(h // 2, h // 2), nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(2, h // 4), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(h + h // 4, h), nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h // 2), nn.SiLU(),
        )
        self.correction_head = nn.Linear(h // 2, 1)
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Identity initialization avoids unsafe steering jumps before training.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, state: torch.Tensor, base_action: torch.Tensor):
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError('Curve adapter state shape mismatch')
        if base_action.ndim != 2 or base_action.shape != (state.shape[0], 2):
            raise ValueError('Curve adapter expects base_action[N,2]')
        ego = self.ego_encoder(state[:, :12].float())
        route = torch.cat((state[:, 12:20].float(), _route_geometry_tensor(state.float())), dim=-1)
        route = self.route_encoder(route)
        action = self.action_encoder(base_action.float())
        features = self.fusion(torch.cat((ego, route, action), dim=-1))
        correction = torch.tanh(self.correction_head(features)).squeeze(-1) * self.config.max_correction
        steering = (base_action[:, 0] + correction).clamp(-1.0, 1.0)
        return steering, correction

    @torch.no_grad()
    def adapt(self, state, base_action, device: str | torch.device = 'cpu'):
        state_np = np.asarray(state, dtype=np.float32)
        base_np = np.asarray(base_action, dtype=np.float32)
        if state_np.shape != (self.config.state_dim,) or base_np.shape != (2,):
            raise ValueError('Curve adapter expects one state vector and [steering,longitudinal] action')
        state_t = torch.as_tensor(state_np[None], dtype=torch.float32, device=device)
        action_t = torch.as_tensor(base_np[None], dtype=torch.float32, device=device)
        steering, correction = self(state_t, action_t)
        adapted = base_np.copy()
        adapted[0] = float(steering[0].cpu())
        corr = float(correction[0].cpu())
        return adapted, {
            'learned_adapter_raw_correction': corr,
            'learned_adapter_blend': min(1.0, abs(corr) / max(self.config.max_correction, 1e-6)),
            'learned_adapter_steering_correction': abs(corr),
            'learned_adapter_base_steering': float(base_np[0]),
            'learned_adapter_applied_steering': float(adapted[0]),
        }


def save_curve_route_adapter(path: str | Path, model: CurveRouteAdapter, *, base_model: str | Path,
                             training: dict[str, Any], dataset: dict[str, np.ndarray] | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'curve_route_adapter_v2',
        'config': asdict(model.config),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'base_model_path': str(base_model),
        'base_model_sha256': sha256(base_model),
        'training': training,
        'dataset_sha256': _dataset_fingerprint(dataset) if dataset is not None else None,
    }
    torch.save(payload, path)
    return {'path': str(path), 'sha256': sha256(path), 'base_model_sha256': payload['base_model_sha256'],
            'dataset_sha256': payload['dataset_sha256']}
