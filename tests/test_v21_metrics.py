import json
from dataclasses import replace
import numpy as np
import pytest
from src.evaluation.metrics import EpisodeMetrics, physical_summary
from src.evaluation.evaluator import evaluate
from src.baselines.random_agent import RandomAgent
from src.utils.config import EnvConfig


def test_time_weighted_p95_does_not_weight_frequency_over_time():
    result = physical_summary([1, 100], [.99, .01], 1)
    assert result['mean_abs'] == pytest.approx(1.99)
    assert result['rms'] == pytest.approx(np.sqrt(100.99))
    assert result['p95_abs'] == 1 and result['peak_abs'] == 100
    assert result['coverage'] == 1


def test_missing_sensors_are_null_not_fabricated_zero_comfort():
    m = EpisodeMetrics(0, 'agent0')
    m.update(1, dict(dt=.1, speed=10, lane_offset=0, jerk=2, lane_width=3.6,
             collision=False, out_of_road=False, success=False, route_completion=.1), False, False, 1)
    row = m.result()
    assert row['lateral_jerk_rms_mps3'] is None
    assert row['lateral_jerk_coverage'] == 0
    assert row['longitudinal_jerk_rms_mps3'] == 2


def test_failure_trace_matches_episode_outcome_and_has_risk_and_reward(tmp_path):
    cfg = EnvConfig(max_steps=8, traffic_vehicles=0)
    report = evaluate(lambda name, seed: RandomAgent(seed), cfg, [3000, 3001], tmp_path,
                      failure_tail_seconds=.3)
    failures = json.loads((tmp_path / 'failure_traces.json').read_text())
    assert len(failures) == report['failure_count'] == 2
    assert report['environment_class'].endswith('NativeWorld')
    assert report['env_config']['reward_version'] == 3
    for failure in failures:
        assert len(failure['trace']) <= 4
        last = failure['trace'][-1]
        assert last['terminated'] or last['truncated']
        assert 'risk' in last and 'reward_terms' in last
        assert last['reward'] == pytest.approx(sum(last['reward_terms'].values()))
    assert report['metrics']['lateral_jerk_coverage'] == pytest.approx(1)
