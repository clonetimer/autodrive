"""Factory for actual shared-world MARL backends (no silent emulation)."""
from src.utils.config import EnvConfig


def make_parallel_env(config: EnvConfig):
    if config.backend == 'native':
        from src.envs.native_world import NativeWorld
        return NativeWorld(config)
    if config.backend == 'metadrive':
        from src.envs.metadrive_wrapper import MetaDriveParallelEnv
        return MetaDriveParallelEnv(config)
    if config.backend == 'replay':
        from src.datasets.replay import ReplayWorld
        return ReplayWorld(config)
    raise ValueError(f'Unsupported backend {config.backend}')


class ParallelDriveEnv:
    """Compatibility constructor; returned backend exposes agents and possible_agents."""
    def __new__(cls, config=None):
        return make_parallel_env(config or EnvConfig(num_agents=2))
