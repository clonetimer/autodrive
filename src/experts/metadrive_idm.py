"""Privileged MetaDrive IDM teacher for *training-only* action labels.

This adapter deliberately exposes the distinction between the learned actor and
an expert that can read MetaDrive's internal navigation/lidar state.  It is used
only to generate imitation targets; deployment/evaluation of a raw learned
policy must not instantiate this class.  The privilege boundary is therefore
explicit and auditable instead of being hidden in the actor observation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExpertDecision:
    steering: float
    longitudinal: float
    action: np.ndarray


class MetaDriveIDMTeacher:
    """Bind MetaDrive's built-in IDM policy to the current learning ego.

    ``MetaDriveParallelEnv`` keeps the authoritative real simulator vehicle.
    The teacher is recreated after every reset because IDM stores routing and
    PID state.  ``target_speed_mps`` is converted to the km/h convention used by
    MetaDrive 0.4.3's IDMPolicy.
    """

    def __init__(self, env, agent_name: str = 'agent0', *, seed: int = 0,
                 target_speed_mps: float | None = None):
        if agent_name not in getattr(env, 'agents', ()):  # fail loudly on stale bindings
            raise ValueError(f'Unknown or inactive MetaDrive agent {agent_name!r}')
        try:
            from metadrive.policy.idm_policy import IDMPolicy
        except ImportError as exc:  # pragma: no cover - exercised by real integration tests
            raise RuntimeError('MetaDrive IDM teacher requires the real MetaDrive runtime') from exc
        raw_id = env._raw_ids[agent_name]
        vehicle = env.env.agents[raw_id]
        self.policy = IDMPolicy(vehicle, int(seed))
        speed = float(target_speed_mps if target_speed_mps is not None else env.config.target_speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('Expert target speed must be positive and finite')
        self.policy.target_speed = speed * 3.6
        self.agent_name = agent_name
        self.seed = int(seed)

    def decide(self) -> ExpertDecision:
        raw = np.asarray(self.policy.act(), dtype=np.float32)
        if raw.shape != (2,) or not np.isfinite(raw).all():
            raise FloatingPointError('MetaDrive IDM teacher produced an invalid action')
        action = np.clip(raw, -1.0, 1.0).astype(np.float32)
        return ExpertDecision(float(action[0]), float(action[1]), action)
