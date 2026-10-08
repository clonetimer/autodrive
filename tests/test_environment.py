from dataclasses import replace
import math
import numpy as np
import pytest
from src.envs.simple_drive_env import SimpleDriveEnv
from src.envs.native_world import NativeWorld
from src.scenarios.traffic import Vehicle, collision
from src.utils.config import EnvConfig


@pytest.mark.parametrize('mode', ['state', 'bev', 'hybrid'])
def test_reset_step_observation_contract(mode):
    env = SimpleDriveEnv(EnvConfig(observation=mode, traffic_vehicles=2))
    obs, info = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    result = env.step(np.zeros(2, dtype=np.float32))
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info['dt'] == 0.1
    assert info['backend'] == 'native'
    env.close()


def test_reset_seed_reproducibility():
    config = EnvConfig(randomize=True, traffic_vehicles=5, observation_noise=0.01)
    a, b = NativeWorld(config), NativeWorld(config)
    oa, _ = a.reset(seed=234)
    ob, _ = b.reset(seed=234)
    np.testing.assert_array_equal(oa['agent0'], ob['agent0'])
    rng = np.random.default_rng(44)
    for _ in range(20):
        action = rng.uniform(-0.3, 0.3, 2).astype(np.float32)
        xa, xb = a.step({'agent0': action}), b.step({'agent0': action})
        np.testing.assert_array_equal(xa[0]['agent0'], xb[0]['agent0'])
        assert xa[1] == xb[1]
    oa, _ = a.reset(seed=234)
    ob, _ = b.reset(seed=234)
    np.testing.assert_array_equal(oa['agent0'], ob['agent0'])


def test_ordinary_reset_advances_rng():
    env = NativeWorld(EnvConfig(traffic_vehicles=0))
    first, _ = env.reset(seed=4)
    second, _ = env.reset()
    assert not np.array_equal(first['agent0'], second['agent0'])


@pytest.mark.parametrize('action', [np.array([np.nan, 0]), [np.inf, 0], [0], [0, 0, 0]])
def test_invalid_action_rejected(world, action):
    with pytest.raises(ValueError):
        world.step({'agent0': action})


def test_out_of_bounds_actions_are_saturated(world):
    obs, *_ = world.step({'agent0': [99, -99]})
    assert np.isfinite(obs['agent0']).all()
    assert abs(world.vehicles['agent0'].steer) <= 0.45


def test_timeout_not_success_or_termination():
    env = NativeWorld(EnvConfig(max_steps=1, road_length=350, traffic_vehicles=0))
    env.reset(seed=0)
    _, _, terms, truncs, infos = env.step({'agent0': [0, 0]})
    assert terms['agent0'] is False
    assert truncs['agent0'] is True
    assert infos['agent0']['success'] is False
    assert env.agents == []
    with pytest.raises(RuntimeError):
        env.step({'agent0': [0, 0]})


def test_collision_distinct_from_offroad(world):
    ego = world.vehicles['agent0']
    world.vehicles['obstacle'] = Vehicle('obstacle', ego.x, ego.y, ego.yaw, 0, 0)
    _, _, term, _, info = world.step({'agent0': [0, -1]})
    assert term['agent0'] and info['agent0']['collision']
    assert not info['agent0']['out_of_road']
    assert not info['agent0']['success']


def test_offroad_is_not_vehicle_collision(world):
    world.vehicles['agent0'].y = 25
    _, _, _, _, infos = world.step({'agent0': [0, 0]})
    assert infos['agent0']['out_of_road']
    assert not infos['agent0']['collision']


def test_progress_and_success_are_physical(world):
    ego = world.vehicles['agent0']
    ego.x = world.road.length - 0.1
    ego.y = float(world.road.lane_y(ego.x, 1))
    _, _, terms, truncs, infos = world.step({'agent0': [0, 0]})
    assert terms['agent0'] and not truncs['agent0']
    assert infos['agent0']['success']
    assert infos['agent0']['route_completion'] == 1
    assert infos['agent0']['progress'] > 0


def test_steering_changes_motion(world):
    yaw = world.vehicles['agent0'].yaw
    world.step({'agent0': [0.5, 0]})
    assert world.vehicles['agent0'].yaw > yaw


def test_oriented_box_collision():
    a = Vehicle('a', 0, 0, 0, 0)
    b = Vehicle('b', 0, 2, math.pi / 2, 0)
    assert collision(a, b)
    b.x = 10
    assert not collision(a, b)


def test_joint_action_keys_must_match(world):
    with pytest.raises(ValueError):
        world.step({})


def test_jerk_has_physical_time_units(world):
    old = world.vehicles['agent0'].acceleration
    _, _, _, _, infos = world.step({'agent0': [0, 0.7]})
    assert infos['agent0']['jerk'] == pytest.approx((infos['agent0']['acceleration'] - old) / 0.1)
