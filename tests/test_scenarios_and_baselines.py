import numpy as np
import pytest
from src.scenarios.generator import split_seeds
from src.scenarios.traffic import idm_acceleration
from src.scenarios.curriculum import Curriculum
from src.baselines.idm_controller import IDMController
from src.baselines.pid_controller import PIDController
from src.envs.native_world import NativeWorld
from src.utils.config import EnvConfig


def test_idm_brakes_when_closing():
    assert idm_acceleration(12, 15, 6, 7) < -1
    assert idm_acceleration(0, 15) > 0
    assert idm_acceleration(10, 0) == -6


def test_seed_partitions_are_disjoint():
    splits = [set(split_seeds(name, 50)) for name in ['train', 'validation', 'test']]
    assert not splits[0] & splits[1]
    assert not splits[0] & splits[2]
    assert not splits[1] & splits[2]
    with pytest.raises(ValueError):
        split_seeds('test', 1000)


def test_curriculum_needs_measured_success():
    curriculum = Curriculum(min_episodes=3)
    assert not curriculum.observe(False, True)
    assert not curriculum.observe(True, False)
    assert not curriculum.observe(True, False)
    assert curriculum.observe(True, False) is False  # 25% collisions > threshold
    for _ in range(40):
        curriculum.observe(True, False)
    assert curriculum.stage >= 1


@pytest.mark.parametrize('controller', [PIDController, IDMController])
@pytest.mark.parametrize('curve', [0, 4])
def test_rule_baseline_completes_straight_and_curved(controller, curve):
    env = NativeWorld(EnvConfig(road_length=70, traffic_vehicles=0, curve_amplitude=curve, max_steps=140))
    obs, _ = env.reset(seed=0)
    policy = controller()
    last = None
    while env.agents:
        obs, _, _, _, info = env.step({'agent0': policy.predict(obs['agent0'])})
        last = info['agent0']
    assert last['success']
    assert not last['collision'] and not last['out_of_road']


def test_idm_lane_change_when_adjacent_clear():
    env = NativeWorld(EnvConfig(scenario='overtake', traffic_vehicles=0, max_steps=220, road_length=140))
    obs, _ = env.reset(seed=0)
    policy = IDMController()
    final = None
    while env.agents:
        obs, _, _, _, infos = env.step({'agent0': policy.predict(obs['agent0'])})
        final = infos['agent0']
    assert final['success']
    assert final['lane_changes'] >= 1
    assert not final['collision']


def test_idm_reduces_speed_for_emergency_leader():
    env = NativeWorld(EnvConfig(scenario='emergency', traffic_vehicles=0, max_steps=100, road_length=300))
    obs, _ = env.reset(seed=0)
    policy = IDMController(allow_lane_change=False)
    speeds, crashes = [], []
    while env.agents:
        obs, _, _, _, info = env.step({'agent0': policy.predict(obs['agent0'])})
        speeds.append(info['agent0']['speed'])
        crashes.append(info['agent0']['collision'])
    assert not any(crashes)
    assert speeds[-1] < max(speeds) * 0.5


def test_random_agent_reset_is_repeatable():
    import numpy as np
    from src.baselines.random_agent import RandomAgent
    agent = RandomAgent(6)
    first = agent.predict(None)
    assert not np.array_equal(first, agent.predict(None))
    agent.reset()
    np.testing.assert_array_equal(first, agent.predict(None))
