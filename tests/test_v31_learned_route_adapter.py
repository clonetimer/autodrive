from pathlib import Path
import numpy as np
import pytest
import torch

from src.navigation.learned_route_adapter import (
    LearnedRouteAdapter, LearnedRouteAdapterAgent, RouteAdapterConfig,
    load_route_adapter, save_route_adapter,
)
from src.evaluation.learned_adapter_metrics import EpisodeMetrics, summarize


def test_learned_route_adapter_starts_near_identity_and_preserves_longitudinal():
    model = LearnedRouteAdapter(RouteAdapterConfig(state_dim=62, hidden_size=32, max_blend=.5))
    state = np.zeros(62, np.float32)
    action = np.array([.35, -.25], np.float32)
    adapted, d = model.adapt(state, action)
    assert adapted[1] == pytest.approx(action[1])
    assert abs(float(adapted[0]) - float(action[0])) < 1e-5
    assert 0 <= d['learned_adapter_blend'] <= .5


def test_learned_route_adapter_shape_validation():
    model = LearnedRouteAdapter(RouteAdapterConfig(state_dim=10, hidden_size=16))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 9), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        model.adapt(np.zeros(9, np.float32), np.zeros(2, np.float32))


def test_route_adapter_artifact_verifies_base_sha(tmp_path):
    base = tmp_path / 'base.pt'
    base.write_bytes(b'base-a')
    model = LearnedRouteAdapter(RouteAdapterConfig(state_dim=8, hidden_size=16))
    path = tmp_path / 'adapter.pt'
    save_route_adapter(path, model, base_model=base, training={'ok': True},
                       dataset={'state': np.zeros((2, 8), np.float32)})
    restored, payload = load_route_adapter(path, base_model=base)
    assert isinstance(restored, LearnedRouteAdapter)
    assert payload['training']['ok'] is True
    other = tmp_path / 'other.pt'
    other.write_bytes(b'base-b')
    with pytest.raises(ValueError, match='different base policy'):
        load_route_adapter(path, base_model=other)


def test_adapter_diagnostics_enter_episode_metrics():
    m = EpisodeMetrics(1, 'agent0')
    info = {
        'dt': .1, 'speed': 5., 'lane_offset': 0., 'lane_width': 3.5, 'jerk': 0.,
        'collision': False, 'out_of_road': False, 'success': False,
        'route_completion': .1, 'progress_m': .5, 'lane_changes': 0,
        'risk': {}, 'route_assist_blend': 0., 'route_assist_need_score': 0.,
        'learned_adapter_blend': .2, 'learned_adapter_steering_correction': .12,
    }
    m.update(1., info, False, False, .2)
    r = m.result()
    assert r['learned_adapter_mean_blend'] == pytest.approx(.2)
    assert r['learned_adapter_mean_steering_correction'] == pytest.approx(.12)
    assert r['learned_adapter_active_time_fraction'] == pytest.approx(1.)
    agg = summarize([r])
    assert agg['learned_adapter_mean_blend'] == pytest.approx(.2)
    assert agg['learned_adapter_peak_steering_correction'] == pytest.approx(.12)
