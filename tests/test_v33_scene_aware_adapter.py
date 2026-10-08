"""Unit contracts for the v3.3 scene-aware temporal adapter."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

from src.navigation.learned_route_adapter import load_route_adapter
from src.navigation.scene_aware_adapter import (
    SceneAwareTemporalAdapter, SceneAwareTemporalAdapterConfig,
    save_scene_aware_adapter)
from src.perception.scene_encoder import SceneEncoder, SceneEncoderConfig


def _history(batch=2, frames=4, neighbors=6):
    dim = 20 + 7 * neighbors
    state = torch.zeros(batch, frames, dim)
    # Plausible route preview.
    route = torch.tensor([.10, .01, .20, .02, .40, .04, .70, .06])
    state[..., 12:20] = route
    for n in range(neighbors):
        start = 20 + 7 * n
        state[..., start:start + 7] = torch.tensor([
            .1 + n * .02, -.05 + n * .01, -.02, .01, 0., 1., 1.])
    return state


def test_scene_aware_adapter_is_exact_identity_at_initialization():
    model = SceneAwareTemporalAdapter(SceneAwareTemporalAdapterConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4))
    history = _history(batch=3, frames=4)
    action = torch.tensor([[.2, .1], [-.3, .4], [0., 0.]])
    steering, correction, risk_logit, attention = model(history, action)
    assert torch.allclose(correction, torch.zeros_like(correction), atol=1e-7)
    assert torch.allclose(steering, action[:, 0], atol=1e-7)
    assert torch.allclose(torch.sigmoid(risk_logit), torch.full_like(risk_logit, .5), atol=1e-7)
    assert torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)


def test_neighbor_set_encoder_is_permutation_invariant():
    torch.manual_seed(7)
    encoder = SceneEncoder(SceneEncoderConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4)).eval()
    history = _history(batch=2, frames=4)
    permuted = history.clone()
    neighbors = history[:, -1, 20:].reshape(2, 6, 7)
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    permuted[:, -1, 20:] = neighbors[:, order].reshape(2, -1)
    with torch.no_grad():
        features_a, _ = encoder(history)
        features_b, _ = encoder(permuted)
    assert torch.allclose(features_a, features_b, atol=2e-6, rtol=2e-6)


def test_temporal_route_encoder_uses_previous_frames():
    torch.manual_seed(11)
    encoder = SceneEncoder(SceneEncoderConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4, use_neighbors=False)).eval()
    a = _history(batch=1, frames=4)
    b = a.clone()
    # Same current observation, different earlier route evolution.
    b[:, :3, 13:20:2] *= -3.0
    with torch.no_grad():
        features_a, _ = encoder(a)
        features_b, _ = encoder(b)
    assert not torch.allclose(features_a, features_b, atol=1e-6, rtol=1e-6)


def test_v33_artifact_loads_through_common_route_adapter_loader(tmp_path):
    base = tmp_path / 'base.bin'
    base.write_bytes(b'base-policy')
    model = SceneAwareTemporalAdapter(SceneAwareTemporalAdapterConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4))
    path = tmp_path / 'adapter.pt'
    save_scene_aware_adapter(path, model, base_model=base, training={'unit': True})
    loaded, payload = load_route_adapter(path, base_model=base)
    assert payload['format'] == 'scene_aware_temporal_adapter_v3'
    assert type(loaded).__name__ == 'SceneAwareTemporalAdapter'
    assert loaded.config.history_len == 4


def test_temporal_history_builder_never_crosses_seed_boundary():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts.train_scene_aware_adapter import _history_tensor

    states = np.arange(5, dtype=np.float32)[:, None]
    seeds = np.array([10, 10, 10, 20, 20])
    steps = np.array([0, 1, 2, 0, 1])
    histories = _history_tensor(states, seeds, steps, history_len=3)[..., 0]
    np.testing.assert_array_equal(histories[0], [0, 0, 0])
    np.testing.assert_array_equal(histories[2], [0, 1, 2])
    np.testing.assert_array_equal(histories[3], [3, 3, 3])
    np.testing.assert_array_equal(histories[4], [3, 3, 4])


def test_common_agent_clears_temporal_history_on_reset(tmp_path, monkeypatch):
    import src.navigation.learned_route_adapter as module

    base = tmp_path / 'base.bin'
    base.write_bytes(b'base-policy')
    model = SceneAwareTemporalAdapter(SceneAwareTemporalAdapterConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4))
    adapter = tmp_path / 'adapter.pt'
    save_scene_aware_adapter(adapter, model, base_model=base, training={'unit': True})

    class FakeBase:
        def __init__(self, checkpoint, device='cpu'):
            self.policy = object()
            self.fingerprint = 'fake'
            self.reset_count = 0
        def reset(self):
            self.reset_count += 1
        def predict(self, state, deterministic=True):
            return np.array([.1, .2], dtype=np.float32)

    monkeypatch.setattr(module, 'EndToEndAgent', FakeBase)
    agent = module.LearnedRouteAdapterAgent(base, adapter)
    state = np.zeros(62, dtype=np.float32)
    state[12:20] = [.1, 0, .2, 0, .4, 0, .7, 0]
    agent.predict(state)
    agent.predict(state)
    assert len(agent._state_history) == 2
    agent.reset()
    assert len(agent._state_history) == 0
