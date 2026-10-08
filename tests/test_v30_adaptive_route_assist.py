import numpy as np
import pytest
from src.navigation.route_assist import RouteAssist, route_assist_need_score
from src.utils.config import EnvConfig
from src.evaluation.metrics import EpisodeMetrics, summarize


def _state(*, lateral_m=0.0, lane_error=0.0, heading_error_rad=0.0):
    state = np.zeros(62, dtype=np.float32)
    state[1] = np.sin(heading_error_rad)
    state[2] = np.cos(heading_error_rad)
    state[3] = lane_error
    for i, forward in enumerate((6.0, 12.0, 24.0, 42.0)):
        state[12 + 2 * i] = forward / 60.0
        state[13 + 2 * i] = lateral_m / 20.0
    return state


def test_adaptive_route_assist_yields_straight_aligned_steering_authority():
    state = _state()
    decision = RouteAssist(1.0, mode='adaptive').decide(np.array([0.0, 0.3], np.float32), state)
    assert decision.need_score == pytest.approx(0.0)
    assert decision.blend == pytest.approx(0.0)
    np.testing.assert_allclose(decision.applied, [0.0, 0.3])
    assert not decision.intervened


def test_adaptive_route_assist_progressively_recovers_authority_on_curve_or_tracking_error():
    easy = route_assist_need_score(_state(lateral_m=0.25))
    curve = route_assist_need_score(_state(lateral_m=3.0))
    drift = route_assist_need_score(_state(lane_error=0.45, heading_error_rad=0.25))
    assert 0.0 <= easy < curve <= 1.0
    assert drift == pytest.approx(1.0)
    decision = RouteAssist(1.0, mode='adaptive').decide(np.array([0.0, -0.2], np.float32), _state(lateral_m=3.0))
    assert decision.blend > 0.0
    assert decision.applied[1] == pytest.approx(-0.2)
    assert decision.intervened


def test_fixed_route_assist_remains_v29_compatible():
    state = _state(lateral_m=2.0)
    action = np.array([0.4, 0.2], np.float32)
    old_default = RouteAssist(0.7).decide(action, state)
    explicit_fixed = RouteAssist(0.7, mode='fixed', min_blend=0.0).decide(action, state)
    np.testing.assert_allclose(old_default.applied, explicit_fixed.applied)
    assert old_default.blend == explicit_fixed.blend == pytest.approx(0.7)
    assert old_default.need_score == explicit_fixed.need_score == pytest.approx(1.0)


def test_adaptive_config_validation_is_fail_closed():
    config = EnvConfig(backend='metadrive', observation='state', route_assist=True,
                       route_assist_mode='adaptive', route_assist_blend=0.9,
                       route_assist_min_blend=0.1)
    assert config.route_assist_mode == 'adaptive'
    with pytest.raises(ValueError):
        EnvConfig(backend='metadrive', observation='state', route_assist=True,
                  route_assist_mode='adaptive', route_assist_blend=0.4,
                  route_assist_min_blend=0.5)
    with pytest.raises(ValueError):
        EnvConfig(backend='metadrive', observation='state', route_assist=True,
                  route_assist_mode='adaptive', route_assist_curvature_start=0.3,
                  route_assist_curvature_full=0.2)


def _metric_info(blend: float, need: float):
    return {'dt': .1, 'speed': 5.0, 'lane_offset': 0.0, 'lane_width': 3.6, 'jerk': 0.0,
            'route_completion': .2, 'collision': False, 'out_of_road': False,
            'success': False, 'progress_m': .5, 'route_assist_intervention': blend > 0,
            'route_assist_steering_correction': .2 * blend, 'route_assist_blend': blend,
            'route_assist_need_score': need, 'safety_intervention': False,
            'safety_severity': 0.0, 'risk': {}}


def test_metrics_report_rule_blend_and_learned_steering_authority_separately():
    metric = EpisodeMetrics(1, 'agent0')
    metric.update(0.0, _metric_info(.75, .8), False, False, .1)
    metric.update(0.0, _metric_info(.25, .3), False, False, .1)
    result = metric.result()
    assert result['route_assist_mean_blend'] == pytest.approx(.5)
    assert result['learned_steering_authority_fraction'] == pytest.approx(.5)
    assert result['route_assist_mean_need_score'] == pytest.approx(.55)
    aggregate = summarize([result])
    assert aggregate['route_assist_mean_blend'] == pytest.approx(.5)
    assert aggregate['learned_steering_authority_fraction'] == pytest.approx(.5)


def test_v30_seed_domains_are_new_and_disjoint_from_v29():
    from src.scenarios.generator import SPLITS, split_seeds
    names = ['route_train', 'validation8', 'release12', 'handoff_train', 'validation9', 'release13']
    domains = [set(range(SPLITS[name][0], SPLITS[name][1] + 1)) for name in names]
    for i, left in enumerate(domains):
        for right in domains[i + 1:]:
            assert left.isdisjoint(right)
    assert split_seeds('validation9', 3) == [29000, 29001, 29002]
    assert split_seeds('release13', 2) == [30000, 30001]
