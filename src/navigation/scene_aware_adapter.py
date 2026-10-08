"""v3.3 scene-aware temporal residual steering adapter.

The deployed component is still a learned residual attached to the frozen v2.8
base actor.  Unlike v3.2 it consumes a short history of policy-visible route
preview and represents surrounding vehicles as a set.  A safety-risk auxiliary
head can be supervised from training-time SafetySupervisor labels, but its
output is diagnostic only and never modifies the deployed action.
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
class SceneAwareTemporalAdapterConfig:
    state_dim: int
    hidden_size: int = 128
    history_len: int = 6
    neighbor_heads: int = 4
    max_correction: float = 0.18
    use_neighbors: bool = True

    def __post_init__(self) -> None:
        SceneEncoderConfig(
            state_dim=self.state_dim,
            hidden_size=self.hidden_size,
            history_len=self.history_len,
            neighbor_heads=self.neighbor_heads,
            use_neighbors=self.use_neighbors,
        )
        if not 0 < self.max_correction <= 0.5:
            raise ValueError('max_correction must lie in (0,0.5]')

    @property
    def neighbors(self) -> int:
        return (self.state_dim - 20) // 7


class SceneAwareTemporalAdapter(nn.Module):
    """Temporal route + interaction-aware direct steering residual."""

    requires_history = True

    def __init__(self, config: SceneAwareTemporalAdapterConfig):
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
        self.correction_head = nn.Linear(h // 2, 1)
        self.risk_head = nn.Linear(h // 2, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Runtime starts as exact identity before training.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        # Neutral p=0.5 is preferable to a fabricated "safe" prior.
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(self, state_history: torch.Tensor, base_action: torch.Tensor):
        if state_history.ndim != 3 or state_history.shape[-1] != self.config.state_dim:
            raise ValueError('Temporal adapter expects state_history[B,T,D]')
        if base_action.ndim != 2 or base_action.shape != (state_history.shape[0], 2):
            raise ValueError('Temporal adapter expects base_action[B,2]')
        scene, attention = self.scene_encoder(state_history)
        action = self.action_encoder(base_action.float())
        features = self.fusion(torch.cat((scene, action), dim=-1))
        correction = torch.tanh(self.correction_head(features)).squeeze(-1) * self.config.max_correction
        steering = (base_action[:, 0] + correction).clamp(-1.0, 1.0)
        risk_logit = self.risk_head(features).squeeze(-1)
        return steering, correction, risk_logit, attention

    @torch.no_grad()
    def adapt(self, state_history, base_action, device: str | torch.device = 'cpu'):
        history = np.asarray(state_history, dtype=np.float32)
        base = np.asarray(base_action, dtype=np.float32)
        if history.ndim == 1:
            history = history[None]
        if history.ndim != 2 or history.shape[1] != self.config.state_dim:
            raise ValueError('Temporal adapter expects history[T,D]')
        if len(history) == 0 or len(history) > self.config.history_len:
            raise ValueError('Temporal adapter received invalid history length')
        if base.shape != (2,) or not np.isfinite(history).all() or not np.isfinite(base).all():
            raise ValueError('Temporal adapter inputs must be finite')
        # Repeat the first available observation at episode start. This avoids a
        # synthetic all-zero state that is outside the trained observation manifold.
        if len(history) < self.config.history_len:
            pad = np.repeat(history[:1], self.config.history_len - len(history), axis=0)
            history = np.concatenate((pad, history), axis=0)
        history_t = torch.as_tensor(history[None], dtype=torch.float32, device=device)
        action_t = torch.as_tensor(base[None], dtype=torch.float32, device=device)
        steering, correction, risk_logit, attention = self(history_t, action_t)
        adapted = base.copy()
        adapted[0] = float(steering[0].cpu())
        corr = float(correction[0].cpu())
        risk = float(torch.sigmoid(risk_logit[0]).cpu())
        attn = attention[0].detach().cpu().numpy()
        current_neighbors = history[-1, 20:].reshape(self.config.neighbors, 7)
        visible = int(np.sum(current_neighbors[:, 6] > 0.5))
        return adapted, {
            'learned_adapter_raw_correction': corr,
            'learned_adapter_blend': min(1.0, abs(corr) / max(self.config.max_correction, 1e-6)),
            'learned_adapter_steering_correction': abs(corr),
            'learned_adapter_base_steering': float(base[0]),
            'learned_adapter_applied_steering': float(adapted[0]),
            'learned_adapter_history_frames': float(min(len(state_history), self.config.history_len)),
            'learned_adapter_risk_probability': risk,
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


def save_scene_aware_adapter(path: str | Path, model: SceneAwareTemporalAdapter, *,
                             base_model: str | Path, training: dict[str, Any],
                             dataset: dict[str, np.ndarray] | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'scene_aware_temporal_adapter_v3',
        'config': asdict(model.config),
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'base_model_path': str(base_model),
        'base_model_sha256': sha256(base_model),
        'training': training,
        'dataset_sha256': _dataset_fingerprint(dataset) if dataset is not None else None,
    }
    torch.save(payload, path)
    return {
        'path': str(path), 'sha256': sha256(path),
        'base_model_sha256': payload['base_model_sha256'],
        'dataset_sha256': payload['dataset_sha256'],
    }
