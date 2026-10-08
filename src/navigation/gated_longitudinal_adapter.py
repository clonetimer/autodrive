"""v4.2 authority-gated longitudinal residual on frozen v3.2 lateral control.

v4.1 showed that longitudinal-only expert transfer can improve interaction
scenarios, but an always-authorized residual occasionally extrapolates to large
braking corrections in ordinary capability scenes (notably with zero visible
neighbours).  v4.2 factorizes *what* correction to make from *whether* the
learned module is allowed to make it:

    v3.2 lateral/base longitudinal -> raw learned longitudinal residual
                                      x structural neighbour-presence gate
                                      x learned intervention-probability gate

Both gates are optional for ablation. Steering is an exact structural
pass-through and SafetySupervisor remains a training-time counterfactual teacher
only; it is never called by the deployment agent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.perception.scene_encoder import SceneEncoder, SceneEncoderConfig
from src.utils.logger import sha256


@dataclass(frozen=True)
class GatedLongitudinalConfig:
    state_dim: int
    hidden_size: int = 128
    history_len: int = 6
    neighbor_heads: int = 4
    use_neighbors: bool = True
    max_longitudinal_correction: float = 1.5
    presence_gate: bool = True
    learned_gate: bool = True
    gate_prior_probability: float = .05

    def __post_init__(self) -> None:
        SceneEncoderConfig(
            state_dim=self.state_dim, hidden_size=self.hidden_size,
            history_len=self.history_len, neighbor_heads=self.neighbor_heads,
            use_neighbors=self.use_neighbors)
        if not 0 < self.max_longitudinal_correction <= 2.0:
            raise ValueError('max_longitudinal_correction must lie in (0,2]')
        if not 0 < self.gate_prior_probability < 1:
            raise ValueError('gate_prior_probability must lie in (0,1)')

    @property
    def neighbors(self) -> int:
        return (self.state_dim - 20) // 7


class GatedLongitudinalTemporalAdapter(nn.Module):
    """Learn a longitudinal residual and an independently supervised authority."""

    requires_history = True

    def __init__(self, config: GatedLongitudinalConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.scene_encoder = SceneEncoder(SceneEncoderConfig(
            state_dim=config.state_dim, hidden_size=h,
            history_len=config.history_len, neighbor_heads=config.neighbor_heads,
            use_neighbors=config.use_neighbors))
        self.action_encoder = nn.Sequential(
            nn.Linear(2, h // 4), nn.LayerNorm(h // 4), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(self.scene_encoder.output_dim + h // 4, h),
            nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h // 2), nn.SiLU())
        self.residual_head = nn.Linear(h // 2, 1)
        self.gate_head = nn.Linear(h // 2, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Residual identity makes the whole adapter a hard no-op at creation.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        # Conservative prior suppresses authority before the gate sees evidence.
        p = self.config.gate_prior_probability
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(np.log(p / (1.0 - p))))

    def _presence(self, state_history: torch.Tensor) -> torch.Tensor:
        latest = state_history[:, -1, 20:].reshape(
            state_history.shape[0], self.config.neighbors, 7)
        return (latest[:, :, 6] > .5).any(dim=1).to(dtype=state_history.dtype)

    def forward(self, state_history: torch.Tensor, lateral_action: torch.Tensor):
        if state_history.ndim != 3 or state_history.shape[-1] != self.config.state_dim:
            raise ValueError('Gated longitudinal adapter expects state_history[B,T,D]')
        if lateral_action.ndim != 2 or lateral_action.shape != (state_history.shape[0], 2):
            raise ValueError('Gated longitudinal adapter expects lateral_action[B,2]')
        scene, attention = self.scene_encoder(state_history)
        action_features = self.action_encoder(lateral_action.float())
        features = self.fusion(torch.cat((scene, action_features), dim=-1))
        raw_delta = torch.tanh(self.residual_head(features)).squeeze(-1)
        raw_delta = raw_delta * self.config.max_longitudinal_correction
        gate_logit = self.gate_head(features).squeeze(-1)
        learned_probability = torch.sigmoid(gate_logit) if self.config.learned_gate else torch.ones_like(gate_logit)
        presence = self._presence(state_history) if self.config.presence_gate else torch.ones_like(raw_delta)
        authority = learned_probability * presence
        effective_delta = raw_delta * authority
        longitudinal = (lateral_action[:, 1] + effective_delta).clamp(-1.0, 1.0)
        action = torch.stack((lateral_action[:, 0], longitudinal), dim=-1)
        return action, raw_delta, effective_delta, authority, learned_probability, gate_logit, presence, attention

    @torch.no_grad()
    def adapt(self, state_history, lateral_action, device: str | torch.device = 'cpu'):
        history = np.asarray(state_history, dtype=np.float32)
        base = np.asarray(lateral_action, dtype=np.float32)
        if history.ndim == 1:
            history = history[None]
        if history.ndim != 2 or history.shape[1] != self.config.state_dim:
            raise ValueError('Gated longitudinal adapter expects history[T,D]')
        if len(history) == 0 or len(history) > self.config.history_len:
            raise ValueError('Gated longitudinal adapter received invalid history length')
        if base.shape != (2,) or not np.isfinite(history).all() or not np.isfinite(base).all():
            raise ValueError('Gated longitudinal adapter inputs must be finite')
        history_frames = len(history)
        if history_frames < self.config.history_len:
            pad = np.repeat(history[:1], self.config.history_len - history_frames, axis=0)
            history = np.concatenate((pad, history), axis=0)
        hist_t = torch.as_tensor(history[None], dtype=torch.float32, device=device)
        action_t = torch.as_tensor(base[None], dtype=torch.float32, device=device)
        action, raw, eff, authority, prob, _, presence, attention = self(hist_t, action_t)
        adapted = action[0].detach().cpu().numpy().astype(np.float32, copy=True)
        if adapted[0] != np.float32(base[0]):
            raise RuntimeError('Gated longitudinal adapter violated exact steering pass-through')
        attn = attention[0].detach().cpu().numpy()
        return adapted, {
            'learned_adapter_longitudinal_correction': abs(float(eff[0].detach().cpu())),
            'learned_adapter_raw_longitudinal_correction': abs(float(raw[0].detach().cpu())),
            'learned_adapter_longitudinal_authority': float(authority[0].detach().cpu()),
            'learned_adapter_gate_probability': float(prob[0].detach().cpu()),
            'learned_adapter_neighbor_presence': float(presence[0].detach().cpu()),
            'learned_adapter_base_longitudinal': float(base[1]),
            'learned_adapter_applied_longitudinal': float(adapted[1]),
            'learned_adapter_history_frames': float(history_frames),
            'learned_adapter_visible_neighbors': float(np.sum(history[-1, 20:].reshape(self.config.neighbors, 7)[:, 6] > .5)),
            'learned_adapter_neighbor_attention_mass': float(attn[:-1].sum()),
        }


def _dataset_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode('utf-8')); digest.update(str(value.shape).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii')); digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def save_gated_longitudinal_adapter(path: str | Path, model: GatedLongitudinalTemporalAdapter, *,
                                    base_model: str | Path, lateral_adapter: str | Path,
                                    training: dict[str, Any],
                                    dataset: dict[str, np.ndarray] | None = None) -> dict:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'gated_longitudinal_temporal_adapter_v42',
        'config': asdict(model.config),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'base_model_path': str(base_model), 'base_model_sha256': sha256(base_model),
        'lateral_adapter_path': str(lateral_adapter), 'lateral_adapter_sha256': sha256(lateral_adapter),
        'training': training,
        'dataset_sha256': _dataset_fingerprint(dataset) if dataset is not None else None,
    }
    torch.save(payload, path)
    return {'path': str(path), 'sha256': sha256(path),
            'base_model_sha256': payload['base_model_sha256'],
            'lateral_adapter_sha256': payload['lateral_adapter_sha256'],
            'dataset_sha256': payload['dataset_sha256']}


def load_gated_longitudinal_adapter(path: str | Path, *, base_model: str | Path,
                                    lateral_adapter: str | Path, device: str = 'cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('format') != 'gated_longitudinal_temporal_adapter_v42':
        raise ValueError('Unsupported gated longitudinal-adapter artifact format')
    if payload.get('base_model_sha256') != sha256(base_model):
        raise ValueError('Gated longitudinal adapter was trained for a different base policy')
    if payload.get('lateral_adapter_sha256') != sha256(lateral_adapter):
        raise ValueError('Gated longitudinal adapter was trained for a different lateral adapter')
    model = GatedLongitudinalTemporalAdapter(GatedLongitudinalConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state_dict'], strict=True); model.eval()
    return model, payload


class GatedFactorizedDrivingAgent:
    """Frozen v3.2 lateral stack + v4.2 authority-gated longitudinal residual."""
    def __init__(self, base_checkpoint: str | Path, lateral_adapter_checkpoint: str | Path,
                 longitudinal_adapter_checkpoint: str | Path, device: str = 'cpu'):
        self.lateral = LearnedRouteAdapterAgent(base_checkpoint, lateral_adapter_checkpoint, device=device)
        self.policy = self.lateral.policy
        self.longitudinal, self.longitudinal_payload = load_gated_longitudinal_adapter(
            longitudinal_adapter_checkpoint, base_model=base_checkpoint,
            lateral_adapter=lateral_adapter_checkpoint, device=device)
        self.device = device
        self.base_fingerprint = self.lateral.base_fingerprint
        self.lateral_fingerprint = sha256(lateral_adapter_checkpoint)
        self.longitudinal_fingerprint = sha256(longitudinal_adapter_checkpoint)
        self.fingerprint = hashlib.sha256(
            (self.base_fingerprint+':'+self.lateral_fingerprint+':'+self.longitudinal_fingerprint).encode('ascii')).hexdigest()
        self._state_history = deque(maxlen=self.longitudinal.config.history_len)
        self.last_diagnostics: dict[str, float] = {}

    def reset(self) -> None:
        self.lateral.reset(); self._state_history.clear(); self.last_diagnostics = {}

    def predict(self, observation, deterministic: bool = True) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float32)
        lateral_action = self.lateral.predict(state, deterministic=deterministic)
        self._state_history.append(state.copy())
        history = np.stack(tuple(self._state_history), axis=0)
        action, long_diag = self.longitudinal.adapt(history, lateral_action, self.device)
        self.last_diagnostics = {**self.lateral.last_diagnostics, **long_diag}
        return action

    def act(self, observations: dict[str, object]) -> dict[str, np.ndarray]:
        return {name: self.predict(obs) for name, obs in observations.items()}
