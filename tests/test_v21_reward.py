"""Reward design regressions are deterministic engineering tests, not safety claims."""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import pytest
from src.envs.native_world import NativeWorld
from src.envs.reward import reward_terms, traffic_risk
from src.scenarios.traffic import Vehicle
from src.utils.config import EnvConfig


def scene(speed=12, target=12):
    world = NativeWorld(EnvConfig(traffic_vehicles=0, target_speed=target, initial_speed=speed))
    world.reset(seed=7)
    ego = world.vehicles['agent0']
    ego.y = 0.0
    return world, world.snapshot('agent0')


@pytest.mark.parametrize('profile', ['balanced', 'safe', 'efficient'])
def test_excess_speed_cannot_buy_more_progress(profile):
    world, snapshot = scene()
    target = sum(reward_terms(snapshot, 1.2, 0, 0.1, profile).values())
    snapshot.ego.speed = 24
    fast = sum(reward_terms(snapshot, 2.4, 0, 0.1, profile).values())
    assert target > fast
    assert reward_terms(snapshot, 2.4, 0, 0.1, profile)['overspeed'] < 0


def test_legacy_bug_is_reproducible_and_not_rewritten():
    world, snapshot = scene()
    normal = sum(reward_terms(snapshot, 1.2, 0, 0.1, version=2).values())
    snapshot.ego.speed = 24
    fast = sum(reward_terms(snapshot, 2.4, 0, 0.1, version=2).values())
    assert fast > normal


@pytest.mark.parametrize('dt', [0.025, 0.05, 0.1, 0.2])
def test_reward_rates_integrate_over_physical_time(dt):
    world, snapshot = scene(15)
    # Same prescribed continuous state/jerk path sampled at different frequencies.
    total = round(2.0 / dt) * sum(reward_terms(snapshot, snapshot.ego.speed * dt, 2.5, dt).values())
    expected = 20 * sum(reward_terms(snapshot, snapshot.ego.speed * .1, 2.5, .1).values())
    assert total == pytest.approx(expected)
    event = reward_terms(snapshot, 0, 0, dt, collision=True)['collision']
    assert event == -12  # terminal event is once-only, not a rate


def test_stopped_leader_allows_needed_stop_not_freezing_on_empty_road():
    world, snapshot = scene(0)
    empty_stop = sum(reward_terms(snapshot, 0, 0, .1).values())
    ego = snapshot.ego
    world.vehicles['leader'] = Vehicle('leader', ego.x + 7, ego.y, 0, 0, 0)
    snapshot = world.snapshot('agent0')
    needed_stop = sum(reward_terms(snapshot, 0, 0, .1).values())
    assert needed_stop > empty_stop
    assert traffic_risk(snapshot).ttc_s is None


def test_same_lane_closing_has_ttc_adjacent_parallel_does_not():
    world, snapshot = scene(12)
    ego = snapshot.ego
    world.vehicles['lead'] = Vehicle('lead', ego.x + 16.5, ego.y, 0, 6)
    risk = traffic_risk(world.snapshot('agent0'))
    assert risk.gap_m == pytest.approx(12)
    assert risk.closing_mps == pytest.approx(6)
    assert risk.ttc_s == pytest.approx(2)
    assert risk.front_risk > 0
    world.vehicles['lead'].y = ego.y + 3.6
    assert traffic_risk(world.snapshot('agent0')).front_id is None


def test_predicted_cut_in_and_fast_rear_approach_are_visible():
    world, snapshot = scene(12)
    ego = snapshot.ego
    world.vehicles['cut'] = Vehicle('cut', ego.x + 12, ego.y + 3.6, -0.2, 12)
    risk = traffic_risk(world.snapshot('agent0'))
    assert risk.front_id == 'cut' and risk.anticipated_overlap
    world.vehicles['cut'].active = False
    world.vehicles['rear'] = Vehicle('rear', ego.x - 12, ego.y, 0, 24)
    assert traffic_risk(world.snapshot('agent0')).rear_risk > 0


def test_near_collision_has_less_reward_than_safe_gap():
    world, snapshot = scene()
    ego = snapshot.ego
    world.vehicles['front'] = Vehicle('front', ego.x + 50, ego.y, 0, 12)
    safe = sum(reward_terms(world.snapshot('agent0'), 1.2, 0, .1).values())
    world.vehicles['front'].x = ego.x + 8
    world.vehicles['front'].speed = 0
    unsafe = sum(reward_terms(world.snapshot('agent0'), 1.2, 0, .1).values())
    assert unsafe < safe


def test_needed_braking_comfort_penalty_does_not_dominate_risk():
    world, snapshot = scene()
    ego = snapshot.ego
    world.vehicles['front'] = Vehicle('front', ego.x + 8, ego.y, 0, 0)
    ego.acceleration = -6
    emergency = reward_terms(world.snapshot('agent0'), 1.0, -50, .1)
    world.vehicles.pop('front')
    unnecessary = reward_terms(world.snapshot('agent0'), 1.0, -50, .1)
    assert abs(emergency['comfort']) < abs(unnecessary['comfort'])


def test_controlled_braking_trajectory_beats_accelerating_into_stopped_leader():
    results = []
    for action in ([0, -1], [0, 1]):
        world, _ = scene(12)
        ego = world.vehicles['agent0']
        world.vehicles['front'] = Vehicle('front', ego.x + 32, ego.y, 0, 0, 0)
        total, collision = 0, False
        for _ in range(30):
            _, r, _, _, i = world.step({'agent0': action})
            total += r['agent0']
            collision |= i['agent0']['collision']
            if not world.agents:
                break
        results.append((total, collision))
    assert results[0][0] > results[1][0]
    assert not results[0][1] and results[1][1]


def test_terminal_success_not_awarded_on_collision():
    _, snapshot = scene()
    assert reward_terms(snapshot, 0, 0, .1, success=True, collision=True)['success'] == 0


def test_step_publishes_reward_and_risk_evidence():
    world, _ = scene()
    _, rewards, _, _, infos = world.step({'agent0': [0, .06]})
    info = infos['agent0']
    assert sum(info['reward_terms'].values()) == pytest.approx(rewards['agent0'])
    assert info['reward_version'] == 3
    assert 'safe_target_mps' in info['risk']
    assert all(math.isfinite(info[k]) for k in ['lateral_acceleration', 'lateral_jerk', 'steering_rate'])
