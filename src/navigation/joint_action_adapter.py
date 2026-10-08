"""v4.0 joint-action temporal residual adapter.

This adapter is the successor to the v3.3 steering-only scene-aware adapter.
It preserves the frozen base actor and policy-visible state contract, but adds a
bounded longitudinal residual head.  The longitudinal head can be disabled for
factorial ablations; when disabled the runtime action is exactly the base
longitudinal command.

No RouteAssist or SafetySupervisor is called at inference.  Those components are
training-time teachers only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.perception.scene_encoder import SceneEncoder, SceneEncoderConfig
from src.utils.logger import sha256


@dataclass(frozen=True)
class JointActionAdapterConfig:
    state_dim: int
    hidden_size: int = 128
    history_len: int = 6
    neighbor_heads: int = 4
    use_neighbors: bool = True
    enable_longitudinal: bool = True
    max_steering_correction: float = 0.18
    max_longitudinal_correction: float = 1.5

    def __post_init__(self) -> None:
        SceneEncoderConfig(
            state_dim=self.state_dim,
            hidden_size=self.hidden_size,
            history_len=self.history_len,
            neighbor_heads=self.neighbor_heads,
            use_neighbors=self.use_neighbors,
        )
        if not 0 < self.max_steering_correction <= 0.5:
            raise ValueError('max_steering_correction must lie in (0,0.5]')
        if not 0 < self.max_longitudinal_correction <= 2.0:
            raise ValueError('max_longitudinal_correction must lie in (0,2]')

    @property
    def neighbors(self) -> int:
        return (self.state_dim - 20) // 7


class JointActionTemporalAdapter(nn.Module):
    """Temporal scene encoder with bounded steering and longitudinal residuals."""

    requires_history = True

    def __init__(self, config: JointActionAdapterConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.scene_encoder = SceneEncoder(SceneEncoderConfig(
            state_dim=config.state_dim,
            hidden_size=h,
            history_len=config.history_len,
            neighbor_heads=config.neighbor_heads,
            use_neighbors=config.use_neighbors,
        ))
        self.action_encoder = nn.Sequential(
            nn.Linear(2, h // 4), nn.LayerNorm(h // 4), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(self.scene_encoder.output_dim + h // 4, h),
            nn.LayerNorm(h), nn.SiLU(),
            nn.Linear(h, h // 2), nn.SiLU(),
        )
        self.steering_head = nn.Linear(h // 2, 1)
        self.longitudinal_head = nn.Linear(h // 2, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Exact identity initialization is a hard runtime contract.
        nn.init.zeros_(self.steering_head.weight)
        nn.init.zeros_(self.steering_head.bias)
        nn.init.zeros_(self.longitudinal_head.weight)
        nn.init.zeros_(self.longitudinal_head.bias)

    def forward(self, state_history: torch.Tensor, base_action: torch.Tensor):
        if state_history.ndim != 3 or state_history.shape[-1] != self.config.state_dim:
            raise ValueError('Joint adapter expects state_history[B,T,D]')
        if base_action.ndim != 2 or base_action.shape != (state_history.shape[0], 2):
            raise ValueError('Joint adapter expects base_action[B,2]')
        scene, attention = self.scene_encoder(state_history)
        action_features = self.action_encoder(base_action.float())
        features = self.fusion(torch.cat((scene, action_features), dim=-1))

        steer_delta = torch.tanh(self.steering_head(features)).squeeze(-1)
        steer_delta = steer_delta * self.config.max_steering_correction
        steering = (base_action[:, 0] + steer_delta).clamp(-1.0, 1.0)

        if self.config.enable_longitudinal:
            long_delta = torch.tanh(self.longitudinal_head(features)).squeeze(-1)
            long_delta = long_delta * self.config.max_longitudinal_correction
            longitudinal = (base_action[:, 1] + long_delta).clamp(-1.0, 1.0)
        else:
            # Keep the graph shape stable while guaranteeing an exact no-op.
            long_delta = torch.zeros_like(steer_delta)
            longitudinal = base_action[:, 1].clamp(-1.0, 1.0)

        action = torch.stack((steering, longitudinal), dim=-1)
        correction = torch.stack((steer_delta, long_delta), dim=-1)
        return action, correction, attention

    @torch.no_grad()
    def adapt(self, state_history, base_action, device: str | torch.device = 'cpu'):
        history = np.asarray(state_history, dtype=np.float32)
        base = np.asarray(base_action, dtype=np.float32)
        if history.ndim == 1:
            history = history[None]
        if history.ndim != 2 or history.shape[1] != self.config.state_dim:
            raise ValueError('Joint adapter expects history[T,D]')
        if len(history) == 0 or len(history) > self.config.history_len:
            raise ValueError('Joint adapter received invalid history length')
        if base.shape != (2,) or not np.isfinite(history).all() or not np.isfinite(base).all():
            raise ValueError('Joint adapter inputs must be finite')
        if len(history) < self.config.history_len:
            pad = np.repeat(history[:1], self.config.history_len - len(history), axis=0)
            history = np.concatenate((pad, history), axis=0)

        hist_t = torch.as_tensor(history[None], dtype=torch.float32, device=device)
        base_t = torch.as_tensor(base[None], dtype=torch.float32, device=device)
        action, correction, attention = self(hist_t, base_t)
        adapted = action[0].detach().cpu().numpy().astype(np.float32, copy=True)
        corr = correction[0].detach().cpu().numpy()
        attn = attention[0].detach().cpu().numpy()
        neighbors = history[-1, 20:].reshape(self.config.neighbors, 7)
        visible = int(np.sum(neighbors[:, 6] > 0.5))
        return adapted, {
            'learned_adapter_raw_correction': float(corr[0]),
            'learned_adapter_blend': min(1.0, abs(float(corr[0])) / max(self.config.max_steering_correction, 1e-6)),
            'learned_adapter_steering_correction': abs(float(corr[0])),
            'learned_adapter_longitudinal_correction': abs(float(corr[1])),
            'learned_adapter_base_steering': float(base[0]),
            'learned_adapter_applied_steering': float(adapted[0]),
            'learned_adapter_base_longitudinal': float(base[1]),
            'learned_adapter_applied_longitudinal': float(adapted[1]),
            'learned_adapter_history_frames': float(min(len(state_history), self.config.history_len)),
            'learned_adapter_visible_neighbors': float(visible),
            'learned_adapter_neighbor_attention_mass': float(attn[:-1].sum()),
            'learned_adapter_longitudinal_enabled': float(self.config.enable_longitudinal),
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


def save_joint_action_adapter(path: str | Path, model: JointActionTemporalAdapter, *,
                              base_model: str | Path, training: dict[str, Any],
                              dataset: dict[str, np.ndarray] | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'joint_action_temporal_adapter_v4',
        'config': asdict(model.config),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'base_model_path': str(base_model),
        'base_model_sha256': sha256(base_model),
        'training': training,
        'dataset_sha256': _dataset_fingerprint(dataset) if dataset is not None else None,
    }
    torch.save(payload, path)
    return {
        'path': str(path),
        'sha256': sha256(path),
        'base_model_sha256': payload['base_model_sha256'],
        'dataset_sha256': payload['dataset_sha256'],
    }
