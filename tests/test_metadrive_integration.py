"""Tests against real MetaDrive are explicitly SKIPPED when unavailable.

No mocks are counted as real simulator integration or real training evidence.
"""
import importlib.util
import numpy as np
import pytest
from src.envs.metadrive_wrapper import prerequisite_status, MetaDriveParallelEnv, MetaDriveUnavailable, validate_action
from src.utils.config import EnvConfig, TrainConfig
from src.utils.spaces import GYMNASIUM_AVAILABLE

READY = prerequisite_status()['ready']


def test_metadrive_action_contract():
    np.testing.assert_array_equal(validate_action([2, -2]), [1, -1])
    with pytest.raises(ValueError):
        validate_action([0, 0, 0])


def test_no_silent_simulator_fallback():
    if READY:
        pytest.skip('Missing-dependency behavior only applies when MetaDrive is absent')
    with pytest.raises(MetaDriveUnavailable, match='Missing dependencies|assets missing'):
        MetaDriveParallelEnv(EnvConfig(backend='metadrive'))


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason='Real Gymnasium is not installed; offline protocol is not substituted in this test')
def test_real_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    from src.envs.simple_drive_env import SimpleDriveEnv
    env = SimpleDriveEnv(EnvConfig(traffic_vehicles=0))
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.metadrive
@pytest.mark.skipif(not READY, reason='Real MetaDrive/Panda3D/Gymnasium/assets not available')
@pytest.mark.parametrize('mode,agents', [('state', 1), ('state', 2), ('bev', 1), ('hybrid', 2)])
def test_real_metadrive_closed_loop_and_training(tmp_path, mode, agents):
    from src.agents.trainer import train
    cfg = TrainConfig(algorithm='mappo' if agents > 1 else 'ppo', total_steps=64,
                      rollout_steps=32, epochs=1, batch_size=16,
                      env=EnvConfig(backend='metadrive', observation=mode, num_agents=agents, max_steps=30))
    result = train(cfg, tmp_path / mode / str(agents))
    assert result['backend'] == 'metadrive'
    assert result['world_steps_total'] == 64
    assert result['parameters_changed_l1'] > 0
