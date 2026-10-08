import numpy as np
from src.envs.native_world import NativeWorld
from src.safety.supervisor import supervise_action, SafetySupervisor
from src.scenarios.traffic import Vehicle
from src.utils.config import EnvConfig


def make_snapshot(gap=12.0, lead_speed=0.0):
    env = NativeWorld(EnvConfig(traffic_vehicles=0, initial_speed=15, target_speed=15))
    env.reset(seed=1)
    ego = env.vehicles['agent0']
    env.vehicles['lead'] = Vehicle('lead', ego.x + gap + (ego.length + 4.5) / 2, ego.y, ego.yaw, lead_speed, lead_speed)
    return env.snapshot('agent0')


def test_supervisor_brakes_short_ttc_without_changing_steering():
    decision = supervise_action(np.array([0.2, 0.8], np.float32), make_snapshot(gap=6, lead_speed=0))
    assert decision.intervened
    assert decision.applied[0] == decision.proposed[0]
    assert decision.applied[1] < 0
    assert decision.reason in {'predictive_front_risk', 'critical_ttc'}


def test_supervisor_is_transparent_on_clear_road():
    env = NativeWorld(EnvConfig(traffic_vehicles=0, initial_speed=10, target_speed=15))
    env.reset(seed=1)
    action = np.array([-0.1, 0.4], np.float32)
    decision = supervise_action(action, env.snapshot('agent0'))
    assert not decision.intervened
    np.testing.assert_array_equal(decision.applied, action)


def test_stateful_supervisor_limits_noncritical_action_chatter():
    supervisor = SafetySupervisor(ttc_threshold_s=1.8, risk_threshold=0.68, max_slew=0.28)
    risky = make_snapshot(gap=18, lead_speed=8)
    first = supervisor.decide(np.array([0.0, 0.85], np.float32), risky)
    second = supervisor.decide(np.array([0.0, 0.85], np.float32), risky)
    assert first.intervened and second.intervened
    assert abs(float(second.applied[1] - first.applied[1])) <= 0.28 + 0.45 * second.severity + 1e-6
    # The filter should not alternate immediately from braking to full throttle.
    assert second.applied[1] < 0.85


def test_stateful_supervisor_recovery_is_rate_limited():
    supervisor = SafetySupervisor(max_slew=0.2, release_steps=1)
    danger = make_snapshot(gap=6, lead_speed=0)
    supervisor.decide(np.array([0.0, 0.8], np.float32), danger)
    clear_env = NativeWorld(EnvConfig(traffic_vehicles=0, initial_speed=8, target_speed=15))
    clear_env.reset(seed=3)
    clear = supervisor.decide(np.array([0.0, 0.8], np.float32), clear_env.snapshot('agent0'))
    assert clear.intervened
    assert clear.reason == 'recovery_slew'
    assert clear.applied[1] < 0.8
