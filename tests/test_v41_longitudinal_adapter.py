"""Unit contracts for the v4.1 factorized longitudinal-only adapter."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from src.navigation.longitudinal_residual_adapter import (
    LongitudinalResidualConfig,
    LongitudinalTemporalAdapter,
    load_longitudinal_adapter,
    save_longitudinal_adapter,
)


def _history(batch=2, frames=6, neighbors=6):
    dim = 20 + 7 * neighbors
    state = torch.zeros(batch, frames, dim)
    state[..., 12:20] = torch.tensor([.1, .01, .2, .02, .4, .03, .7, .05])
    for n in range(neighbors):
        start = 20 + 7 * n
        state[..., start:start+7] = torch.tensor([.1+n*.03, .02*n, -.01, .0, 0., 1., 1.])
    return state


def test_v41_longitudinal_adapter_is_exact_identity_at_initialization():
    model = LongitudinalTemporalAdapter(LongitudinalResidualConfig(
        62, hidden_size=64, history_len=6, neighbor_heads=4))
    base = torch.tensor([[.2, .4], [-.3, -.2], [0., 1.]])
    action, correction, attention = model(_history(batch=3), base)
    assert torch.equal(action[:, 0], base[:, 0])
    assert torch.allclose(action[:, 1], base[:, 1], atol=1e-7)
    assert torch.allclose(correction, torch.zeros(3), atol=1e-7)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)


def test_v41_steering_is_structurally_passthrough_after_long_head_changes():
    model = LongitudinalTemporalAdapter(LongitudinalResidualConfig(
        62, hidden_size=64, history_len=6, neighbor_heads=4))
    with torch.no_grad():
        model.longitudinal_head.weight.fill_(.17)
        model.longitudinal_head.bias.fill_(.25)
    base = torch.tensor([[.12345679, .4], [-.7654321, -.2]], dtype=torch.float32)
    action, correction, _ = model(_history(batch=2), base)
    # Stronger than allclose: the forward graph never computes a replacement steering value.
    assert torch.equal(action[:, 0], base[:, 0])
    assert torch.any(correction.abs() > 0)


def test_v41_artifact_binds_frozen_lateral_sha(tmp_path):
    base = tmp_path / 'base.bin'; base.write_bytes(b'base-policy')
    lateral = tmp_path / 'lateral.pt'; lateral.write_bytes(b'v32-lateral')
    wrong_lateral = tmp_path / 'other-lateral.pt'; wrong_lateral.write_bytes(b'other')
    model = LongitudinalTemporalAdapter(LongitudinalResidualConfig(
        62, hidden_size=64, history_len=6, neighbor_heads=4))
    artifact = tmp_path / 'long.pt'
    meta = save_longitudinal_adapter(
        artifact, model, base_model=base, lateral_adapter=lateral,
        training={'unit': True})
    assert meta['lateral_adapter_sha256']
    loaded, payload = load_longitudinal_adapter(
        artifact, base_model=base, lateral_adapter=lateral)
    assert payload['format'] == 'longitudinal_temporal_adapter_v41'
    assert isinstance(loaded, LongitudinalTemporalAdapter)
    with pytest.raises(ValueError, match='different lateral adapter'):
        load_longitudinal_adapter(
            artifact, base_model=base, lateral_adapter=wrong_lateral)


def test_v41_adapt_reports_exact_steering_passthrough():
    model = LongitudinalTemporalAdapter(LongitudinalResidualConfig(
        62, hidden_size=64, history_len=6, neighbor_heads=4))
    with torch.no_grad():
        model.longitudinal_head.bias.fill_(.3)
    hist = _history(batch=1)[0].numpy()
    lateral = np.asarray([.2345679, .1], dtype=np.float32)
    action, diag = model.adapt(hist, lateral)
    assert action[0] == lateral[0]
    assert diag['learned_adapter_longitudinal_correction'] > 0


def test_v41_history_builder_never_crosses_seed_boundary():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts.train_longitudinal_adapter import _history_tensor
    states = np.arange(5, dtype=np.float32)[:, None]
    seeds = np.array([10, 10, 10, 20, 20])
    steps = np.array([0, 1, 2, 0, 1])
    h = _history_tensor(states, seeds, steps, 3)[..., 0]
    np.testing.assert_array_equal(h[0], [0, 0, 0])
    np.testing.assert_array_equal(h[2], [0, 1, 2])
    np.testing.assert_array_equal(h[3], [3, 3, 3])
    np.testing.assert_array_equal(h[4], [3, 3, 4])


def test_v41_collection_schedule_is_protected_then_raw_for_mixed():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts.train_longitudinal_adapter import _environment_safety_for_iteration
    assert [_environment_safety_for_iteration('protected', i) for i in range(3)] == [True, True, True]
    assert [_environment_safety_for_iteration('mixed', i) for i in range(3)] == [True, False, False]
    assert [_environment_safety_for_iteration('raw', i) for i in range(3)] == [False, False, False]
