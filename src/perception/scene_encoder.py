"""Compact scene encoder for state-based autonomous-driving policies.

The encoder keeps the v2/v3 policy-visible observation contract intact while
adding two inductive biases that the flattened MLP adapters do not have:

* ordered route preview is encoded over a short temporal window;
* surrounding vehicles are treated as a masked set with attention rather than
  as six semantically meaningful fixed slots.

No simulator geometry, map API, RouteAssist controller, or privileged expert
state is consumed here.  All inputs are slices of the policy-visible state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class SceneEncoderConfig:
    state_dim: int
    hidden_size: int = 128
    history_len: int = 6
    neighbor_heads: int = 4
    use_neighbors: bool = True

    def __post_init__(self) -> None:
        if self.state_dim < 27 or (self.state_dim - 20) % 7:
            raise ValueError('Scene encoder requires state_dim=20+7*N with at least one neighbor slot')
        if self.hidden_size < 32 or self.hidden_size % 4:
            raise ValueError('hidden_size must be >=32 and divisible by 4')
        if not 1 <= self.history_len <= 32:
            raise ValueError('history_len must lie in [1,32]')
        token_dim = self.hidden_size // 2
        if self.neighbor_heads <= 0 or token_dim % self.neighbor_heads:
            raise ValueError('neighbor_heads must divide hidden_size/2')

    @property
    def neighbors(self) -> int:
        return (self.state_dim - 20) // 7

    @property
    def token_dim(self) -> int:
        return self.hidden_size // 2


def route_geometry_sequence(state_history: torch.Tensor) -> torch.Tensor:
    """Return normalized route heading and heading-difference features.

    Args:
        state_history: ``[B,T,D]`` policy-visible state history.

    Returns:
        ``[B,T,8]`` features derived only from state[12:20].
    """
    if state_history.ndim != 3 or state_history.shape[-1] < 20:
        raise ValueError('route geometry expects state history[B,T,D>=20]')
    route = state_history[..., 12:20].reshape(*state_history.shape[:2], 4, 2)
    x = route[..., 0] * 60.0
    y = route[..., 1] * 20.0
    # atan2 is deliberately used directly; clamping x would erase the sign on
    # severe bends where a preview point can move behind the ego frame.
    angles = torch.atan2(y, x)
    delta = torch.zeros_like(angles)
    delta[..., 0] = angles[..., 0]
    delta[..., 1:] = angles[..., 1:] - angles[..., :-1]
    # Wrap differences to avoid a discontinuity around +/-pi.
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.cat((angles / 0.7, delta / 0.35), dim=-1).clamp(-5.0, 5.0)


class SceneEncoder(nn.Module):
    """Encode current ego state, temporal route intent and neighbor set."""

    def __init__(self, config: SceneEncoderConfig):
        super().__init__()
        self.config = config
        d = config.token_dim

        self.ego_encoder = nn.Sequential(
            nn.Linear(12, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.route_frame_encoder = nn.Sequential(
            nn.Linear(16, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.route_gru = nn.GRU(d, d, batch_first=True)

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(7, d), nn.LayerNorm(d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=d, num_heads=config.neighbor_heads, batch_first=True)
        # A permanently valid null token avoids NaNs when no neighbors are
        # visible. It also gives attention an explicit "no interaction" option.
        self.null_neighbor = nn.Parameter(torch.zeros(1, 1, d))

    @property
    def output_dim(self) -> int:
        # ego + temporal route + neighbor context
        return 3 * self.config.token_dim

    def _validate(self, state_history: torch.Tensor) -> None:
        if state_history.ndim != 3:
            raise ValueError('Scene encoder expects state_history[B,T,D]')
        if state_history.shape[-1] != self.config.state_dim:
            raise ValueError('Scene encoder state dimension mismatch')
        if not 1 <= state_history.shape[1] <= self.config.history_len:
            raise ValueError('Scene encoder history length exceeds configured window')

    def forward(self, state_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(state_history)
        history = state_history.float()
        current = history[:, -1]
        ego = self.ego_encoder(current[:, :12])

        route = torch.cat((history[..., 12:20], route_geometry_sequence(history)), dim=-1)
        route_frames = self.route_frame_encoder(route)
        _, route_hidden = self.route_gru(route_frames)
        route_context = route_hidden[-1]

        batch = history.shape[0]
        if self.config.use_neighbors:
            neighbors = current[:, 20:].reshape(batch, self.config.neighbors, 7)
            visible = neighbors[..., 6] > 0.5
            neighbor_tokens = self.neighbor_encoder(neighbors)
            null = self.null_neighbor.expand(batch, -1, -1)
            keys = torch.cat((neighbor_tokens, null), dim=1)
            padding = torch.cat(
                ((~visible), torch.zeros(batch, 1, dtype=torch.bool, device=history.device)), dim=1)
            neighbor_context, attention = self.neighbor_attention(
                ego[:, None], keys, keys, key_padding_mask=padding,
                need_weights=True, average_attn_weights=True)
            neighbor_context = neighbor_context[:, 0]
            attention = attention[:, 0]
        else:
            neighbor_context = torch.zeros_like(ego)
            # Keep diagnostics shape stable: N neighbor slots + null slot.
            attention = torch.zeros(
                batch, self.config.neighbors + 1, dtype=history.dtype, device=history.device)
            attention[:, -1] = 1.0

        features = torch.cat((ego, route_context, neighbor_context), dim=-1)
        return features, attention
