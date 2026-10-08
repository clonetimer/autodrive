"""Gymnasium single-agent facade over the native coupled-traffic world."""
from dataclasses import replace
from src.envs.native_world import NativeWorld
from src.utils.config import EnvConfig
from src.utils.spaces import Env


class SimpleDriveEnv(Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, config: EnvConfig | None = None, max_steps: int | None = None,
                 seed: int | None = None, render_mode: str | None = None):
        super().__init__()
        config = config or EnvConfig()
        updates = {'num_agents': 1}
        if max_steps is not None:
            updates['max_steps'] = max_steps
        if seed is not None:
            updates['seed'] = seed
        self.world = NativeWorld(replace(config, **updates))
        self.observation_space = self.world.observation_space
        self.action_space = self.world.action_space
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, infos = self.world.reset(seed=seed, options=options)
        return observations['agent0'], infos['agent0']

    def step(self, action):
        o, r, t, u, i = self.world.step({'agent0': action})
        return o['agent0'], r['agent0'], t['agent0'], u['agent0'], i['agent0']

    def render(self):
        return self.world.render()

    def close(self):
        self.world.close()
