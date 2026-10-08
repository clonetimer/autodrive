import pytest
from src.envs.metadrive_wrapper import prerequisite_status

READY = prerequisite_status()['ready']


@pytest.mark.metadrive
@pytest.mark.skipif(not READY, reason='Real MetaDrive/Panda3D/Gymnasium/assets not available')
def test_official_metadrive_gymnasium_compatibility():
    from gymnasium.utils.env_checker import check_env
    from src.envs.metadrive_official import OfficialMetaDriveGymEnv
    from src.envs.metadrive_contract import reference_config

    env = OfficialMetaDriveGymEnv(reference_config(start_seed=0, num_scenarios=2, horizon=20))
    try:
        check_env(env, skip_render_check=True, skip_close_check=True)
        obs, info = env.reset(seed=0, options={})
        assert env.observation_space.contains(obs)
        result = env.step(env.action_space.sample())
        assert len(result) == 5
    finally:
        env.close()
