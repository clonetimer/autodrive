"""Deployment: observation -> trainable encoder -> actor -> normalized controls.

The actor itself contains no rule controller. Environments may optionally apply
a separately reported longitudinal ``SafetySupervisor`` after the actor; raw and
supervised metrics must therefore be reported as different systems. Supported
learned inputs include state, semantic BEV and real MetaDrive RGB plus explicit
ego/navigation motion. ``rgb_ego`` is not camera-only perception because motion
and navigation signals are fused with pixels.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.agents.trainer import load_checkpoint
from src.utils.logger import sha256


class EndToEndAgent:
    def __init__(self, checkpoint: str | Path, device: str = 'cpu'):
        self.policy, self.config, self.checkpoint = load_checkpoint(checkpoint, device)
        self.fingerprint = sha256(checkpoint)

    def reset(self) -> None:
        """Stateless feed-forward policy: no history leaks across episodes."""

    def predict(self, observation, deterministic: bool = True) -> np.ndarray:
        action = self.policy.predict(observation, deterministic)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise FloatingPointError('Policy produced an invalid vehicle command')
        return action

    def act(self, observations: dict[str, object]) -> dict[str, np.ndarray]:
        # Shared actor; no centralized critic information is used at deployment.
        return {name: self.predict(observation) for name, observation in observations.items()}
