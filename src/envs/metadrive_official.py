"""Gymnasium compatibility facade for MetaDrive 0.4.3 official observation/reward.

MetaDrive 0.4.3 already returns Gymnasium-style reset/step values, but its reset
signature does not accept Gymnasium's ``options`` keyword. Modern Gymnasium/SB3
call ``reset(seed=..., options=...)``. This small facade fixes only that API
boundary and deliberately leaves MetaDrive observations, rewards and dynamics
untouched.
"""
from __future__ import annotations
from typing import Any
from src.utils.spaces import Env


class OfficialMetaDriveGymEnv(Env):
    """Expose a raw MetaDriveEnv through the modern Gymnasium reset contract."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]):
        try:
            from metadrive import MetaDriveEnv
        except ImportError as exc:  # pragma: no cover - exercised by optional integration gate
            raise RuntimeError("MetaDrive is required for OfficialMetaDriveGymEnv") from exc
        self.raw_env = MetaDriveEnv(config)
        self.action_space = self.raw_env.action_space
        self.observation_space = self.raw_env.observation_space
        self.metadata = dict(getattr(self.raw_env, "metadata", self.metadata))
        self.render_mode = None
        try:
            from gymnasium.envs.registration import EnvSpec
            self.spec = EnvSpec(
                id='MetaDriveOfficialCompat-v0', entry_point=None,
                nondeterministic=True, max_episode_steps=int(config.get('horizon', 0)) or None,
            )
        except ImportError:
            self.spec = None
        self.start_seed = int(config.get('start_seed', 0))
        self.num_scenarios = int(config.get('num_scenarios', 1))
        if self.num_scenarios <= 0:
            raise ValueError('Official procedural compatibility requires num_scenarios > 0')
        self.last_scenario_seed = self.start_seed

    @property
    def unwrapped(self):
        # For Gymnasium diagnostics, this compatibility facade is the public env.
        return self

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Initialize Gymnasium's RNG contract while keeping MetaDrive's official
        # reset implementation and scenario seed semantics unchanged.
        super().reset(seed=seed)
        if options not in (None, {}):
            raise ValueError("MetaDrive 0.4.3 has no reset(options=...) semantics")
        scenario_seed = self.start_seed if seed is None else self.start_seed + (int(seed) % self.num_scenarios)
        self.last_scenario_seed = scenario_seed
        return self.raw_env.reset(seed=scenario_seed)

    def step(self, action):
        return self.raw_env.step(action)

    def render(self):
        return self.raw_env.render()

    def close(self):
        return self.raw_env.close()

    def __getattr__(self, name: str):
        # Useful for instrumentation such as vehicle.speed without changing the
        # official observation/reward stream.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.raw_env, name)
