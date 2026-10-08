import pytest
from src.evaluation.metrics import EpisodeMetrics


def test_safety_metrics_report_intervention_magnitude_and_severity():
    metric = EpisodeMetrics(1, 'agent0')
    base = dict(dt=.1, speed=10., lane_offset=0., lane_width=3.6, jerk=0., collision=False,
                out_of_road=False, success=False, route_completion=.2, lane_changes=0,
                lateral_acceleration=0., lateral_jerk=0., steering_rate=0., risk={})
    metric.update(0., dict(base, safety_intervention=True, safety_severity=.8,
                           proposed_action=[0., .8], applied_action=[0., -.2]), False, False, .1)
    metric.update(0., dict(base, safety_intervention=False, safety_severity=.1,
                           proposed_action=[0., .2], applied_action=[0., .2]), False, True, .1)
    result = metric.result()
    assert result['safety_intervention_time_fraction'] == .5
    assert result['safety_mean_longitudinal_correction'] == .5
    assert result['safety_peak_longitudinal_correction'] == 1.0
    assert result['safety_mean_severity'] == pytest.approx(.45)
