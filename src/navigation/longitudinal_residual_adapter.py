"""v4.1 factorized longitudinal residual on top of the frozen v3.2 lateral policy.

The registered v3.2 curve-route adapter remains solely responsible for steering.
This module learns only a bounded longitudinal residual from temporal ego/route
state and permutation-invariant neighbour features.  It binds both the frozen
base PPO SHA and the frozen v3.2 lateral-adapter SHA into its artifact.

No SafetySupervisor is called at deployment.  The supervisor is a training-time
counterfactual teacher only.
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
class LongitudinalResidualConfig:
    state_dim: int
    hidden_size: int = 128
    history_len: int = 6
    neighbor_heads: int = 4
    use_neighbors: bool = True
    max_longitudinal_correction: float = 1.5

    def __post_init__(self) -> None:
        SceneEncoderConfig(
            state_dim=self.state_dim, hidden_size=self.hidden_size,
            history_len=self.history_len, neighbor_heads=self.neighbor_heads,
            use_neighbors=self.use_neighbors)
        if not 0 < self.max_longitudinal_correction <= 2.0:
            raise ValueError('max_longitudinal_correction must lie in (0,2]')

    @property
    def neighbors(self) -> int:
        return (self.state_dim - 20) // 7


class LongitudinalTemporalAdapter(nn.Module):
    """Temporal scene encoder with an exact steering pass-through."""

    requires_history = True

    def __init__(self, config: LongitudinalResidualConfig):
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
            nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h // 2), nn.SiLU())
        self.longitudinal_head = nn.Linear(h // 2, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Hard identity initialization: the frozen v3.2 action is unchanged.
        nn.init.zeros_(self.longitudinal_head.weight)
        nn.init.zeros_(self.longitudinal_head.bias)

    def forward(self, state_history: torch.Tensor, lateral_action: torch.Tensor):
        if state_history.ndim != 3 or state_history.shape[-1] != self.config.state_dim:
            raise ValueError('Longitudinal adapter expects state_history[B,T,D]')
        if lateral_action.ndim != 2 or lateral_action.shape != (state_history.shape[0], 2):
            raise ValueError('Longitudinal adapter expects lateral_action[B,2]')
        scene, attention = self.scene_encoder(state_history)
        action_features = self.action_encoder(lateral_action.float())
        features = self.fusion(torch.cat((scene, action_features), dim=-1))
        delta = torch.tanh(self.longitudinal_head(features)).squeeze(-1)
        delta = delta * self.config.max_longitudinal_correction
        longitudinal = (lateral_action[:, 1] + delta).clamp(-1.0, 1.0)
        # Exact steering pass-through is a structural contract, not a loss term.
        action = torch.stack((lateral_action[:, 0], longitudinal), dim=-1)
        return action, delta, attention

    @torch.no_grad()
    def adapt(self, state_history, lateral_action, device: str | torch.device = 'cpu'):
        history = np.asarray(state_history, dtype=np.float32)
        base = np.asarray(lateral_action, dtype=np.float32)
        if history.ndim == 1:
            history = history[None]
        if history.ndim != 2 or history.shape[1] != self.config.state_dim:
            raise ValueError('Longitudinal adapter expects history[T,D]')
        if len(history) == 0 or len(history) > self.config.history_len:
            raise ValueError('Longitudinal adapter received invalid history length')
        if base.shape != (2,) or not np.isfinite(history).all() or not np.isfinite(base).all():
            raise ValueError('Longitudinal adapter inputs must be finite')
        history_frames = len(history)
        if history_frames < self.config.history_len:
            pad = np.repeat(history[:1], self.config.history_len - history_frames, axis=0)
            history = np.concatenate((pad, history), axis=0)
        hist_t = torch.as_tensor(history[None], dtype=torch.float32, device=device)
        action_t = torch.as_tensor(base[None], dtype=torch.float32, device=device)
        action, delta, attention = self(hist_t, action_t)
        adapted = action[0].detach().cpu().numpy().astype(np.float32, copy=True)
        corr = float(delta[0].detach().cpu())
        attn = attention[0].detach().cpu().numpy()
        neighbors = history[-1, 20:].reshape(self.config.neighbors, 7)
        visible = int(np.sum(neighbors[:, 6] > 0.5))
        if adapted[0] != np.float32(base[0]):
            raise RuntimeError('Longitudinal adapter violated exact steering pass-through')
        return adapted, {
            'learned_adapter_longitudinal_correction': abs(corr),
            'learned_adapter_base_longitudinal': float(base[1]),
            'learned_adapter_applied_longitudinal': float(adapted[1]),
            'learned_adapter_history_frames': float(history_frames),
            'learned_adapter_visible_neighbors': float(visible),
            'learned_adapter_neighbor_attention_mass': float(attn[:-1].sum()),
        }


def _dataset_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode('utf-8'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def save_longitudinal_adapter(path: str | Path, model: LongitudinalTemporalAdapter, *,
                              base_model: str | Path, lateral_adapter: str | Path,
                              training: dict[str, Any],
                              dataset: dict[str, np.ndarray] | None = None) -> dict:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'longitudinal_temporal_adapter_v41',
        'config': asdict(model.config),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'base_model_path': str(base_model),
        'base_model_sha256': sha256(base_model),
        'lateral_adapter_path': str(lateral_adapter),
        'lateral_adapter_sha256': sha256(lateral_adapter),
        'training': training,
        'dataset_sha256': _dataset_fingerprint(dataset) if dataset is not None else None,
    }
    torch.save(payload, path)
    return {'path': str(path), 'sha256': sha256(path),
            'base_model_sha256': payload['base_model_sha256'],
            'lateral_adapter_sha256': payload['lateral_adapter_sha256'],
            'dataset_sha256': payload['dataset_sha256']}


def load_longitudinal_adapter(path: str | Path, *, base_model: str | Path,
                              lateral_adapter: str | Path, device: str = 'cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('format') != 'longitudinal_temporal_adapter_v41':
        raise ValueError('Unsupported longitudinal-adapter artifact format')
    if payload.get('base_model_sha256') != sha256(base_model):
        raise ValueError('Longitudinal adapter was trained for a different base policy')
    if payload.get('lateral_adapter_sha256') != sha256(lateral_adapter):
        raise ValueError('Longitudinal adapter was trained for a different lateral adapter')
    model = LongitudinalTemporalAdapter(LongitudinalResidualConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model, payload


class FactorizedDrivingAgent:
    """Frozen v3.2 lateral policy + learned longitudinal-only residual."""

    def __init__(self, base_checkpoint: str | Path, lateral_adapter_checkpoint: str | Path,
                 longitudinal_adapter_checkpoint: str | Path, device: str = 'cpu'):
        self.lateral = LearnedRouteAdapterAgent(base_checkpoint, lateral_adapter_checkpoint, device=device)
        self.policy = self.lateral.policy
        self.longitudinal, self.longitudinal_payload = load_longitudinal_adapter(
            longitudinal_adapter_checkpoint, base_model=base_checkpoint,
            lateral_adapter=lateral_adapter_checkpoint, device=device)
        self.device = device
        self.base_fingerprint = self.lateral.base_fingerprint
        self.lateral_fingerprint = sha256(lateral_adapter_checkpoint)
        self.longitudinal_fingerprint = sha256(longitudinal_adapter_checkpoint)
        self.fingerprint = hashlib.sha256(
            (self.base_fingerprint + ':' + self.lateral_fingerprint + ':' + self.longitudinal_fingerprint).encode('ascii')).hexdigest()
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
        # Preserve the v3.2 lateral diagnostics, add only longitudinal/scene data.
        self.last_diagnostics = {**self.lateral.last_diagnostics, **long_diag}
        return action

    def act(self, observations: dict[str, object]) -> dict[str, np.ndarray]:
        return {name: self.predict(obs) for name, obs in observations.items()}
