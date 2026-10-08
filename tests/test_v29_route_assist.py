import numpy as np
import pytest
from src.navigation.route_assist import RouteAssist, pure_pursuit_steering
from src.utils.config import EnvConfig
from src.evaluation.metrics import EpisodeMetrics


def _state(lateral=0.0):
    s=np.zeros(62,dtype=np.float32);s[12]=8/60;s[13]=lateral/20;s[14]=16/60;s[16]=30/60;s[18]=50/60
    return s


def test_route_assist_is_transparent_and_only_changes_steering():
    d=RouteAssist(1.0).decide(np.array([.7,.25],np.float32),_state(2.0))
    assert d.intervened
    assert d.applied[1] == pytest.approx(.25)
    assert d.applied[0] == pytest.approx(d.target_steering)
    assert d.correction > 0


def test_route_assist_zero_blend_is_identity():
    d=RouteAssist(0.0).decide(np.array([.2,-.3],np.float32),_state(3.0))
    np.testing.assert_allclose(d.applied,[.2,-.3])
    assert not d.intervened


def test_route_assist_config_requires_metadrive_state():
    with pytest.raises(ValueError): EnvConfig(route_assist=True)
    with pytest.raises(ValueError): EnvConfig(backend='metadrive',observation='bev_ego',route_assist=True)
    assert EnvConfig(backend='metadrive',observation='state',route_assist=True).route_assist


def test_route_assist_metrics_are_reported_separately_from_longitudinal_safety():
    m=EpisodeMetrics(1,'agent0')
    info={'dt':.1,'speed':5.0,'lane_offset':0.0,'lane_width':3.6,'jerk':0.0,'route_completion':.2,
          'collision':False,'out_of_road':False,'success':False,'progress_m':.5,
          'route_assist_intervention':True,'route_assist_steering_correction':.4,
          'safety_intervention':False,'safety_severity':0.0,'risk':{}}
    m.update(1.0,info,False,False,.1)
    r=m.result()
    assert r['route_assist_intervention_time_fraction'] == pytest.approx(1.0)
    assert r['route_assist_mean_steering_correction'] == pytest.approx(.4)
    assert r['safety_intervention_time_fraction'] == 0
