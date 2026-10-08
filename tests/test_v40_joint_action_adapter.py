"""Unit contracts for the v4.0 joint-action residual adapter."""
from pathlib import Path
import sys

import numpy as np
import torch

from src.navigation.joint_action_adapter import (
    JointActionAdapterConfig, JointActionTemporalAdapter,
    save_joint_action_adapter)
from src.navigation.learned_route_adapter import load_route_adapter


def _history(batch=2, frames=4, neighbors=6):
    dim = 20 + 7 * neighbors
    state = torch.zeros(batch, frames, dim)
    state[..., 12:20] = torch.tensor([.1,.01,.2,.02,.4,.03,.7,.05])
    for n in range(neighbors):
        start = 20 + 7*n
        state[..., start:start+7] = torch.tensor([.1+n*.03, .02*n, -.01, .0, 0., 1., 1.])
    return state


def test_v40_joint_adapter_is_exact_identity_at_initialization():
    model = JointActionTemporalAdapter(JointActionAdapterConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4,
        enable_longitudinal=True))
    history = _history(batch=3)
    base = torch.tensor([[.2,.4],[-.3,-.2],[0.,1.]])
    action, correction, attention = model(history, base)
    assert torch.allclose(action, base, atol=1e-7)
    assert torch.allclose(correction, torch.zeros_like(correction), atol=1e-7)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)


def test_v40_disabled_longitudinal_is_hard_noop_even_if_head_weights_change():
    model = JointActionTemporalAdapter(JointActionAdapterConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4,
        enable_longitudinal=False))
    with torch.no_grad():
        model.longitudinal_head.weight.fill_(9.0)
        model.longitudinal_head.bias.fill_(-4.0)
    base = torch.tensor([[.1,.71],[-.2,-.55]])
    action, correction, _ = model(_history(batch=2), base)
    assert torch.allclose(action[:,1], base[:,1], atol=1e-7)
    assert torch.allclose(correction[:,1], torch.zeros(2), atol=1e-7)


def test_v40_artifact_loads_through_common_adapter_loader(tmp_path):
    base = tmp_path/'base.bin'; base.write_bytes(b'base')
    model = JointActionTemporalAdapter(JointActionAdapterConfig(
        62, hidden_size=64, history_len=4, neighbor_heads=4))
    path = tmp_path/'joint.pt'
    save_joint_action_adapter(path, model, base_model=base, training={'unit':True})
    loaded, payload = load_route_adapter(path, base_model=base)
    assert payload['format'] == 'joint_action_temporal_adapter_v4'
    assert type(loaded).__name__ == 'JointActionTemporalAdapter'
    assert loaded.config.enable_longitudinal is True


def test_v40_history_builder_never_crosses_seed_boundary():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts.train_joint_action_adapter import _history_tensor
    states=np.arange(5,dtype=np.float32)[:,None]
    seeds=np.array([10,10,10,20,20]); steps=np.array([0,1,2,0,1])
    h=_history_tensor(states,seeds,steps,3)[...,0]
    np.testing.assert_array_equal(h[0],[0,0,0])
    np.testing.assert_array_equal(h[2],[0,1,2])
    np.testing.assert_array_equal(h[3],[3,3,3])
    np.testing.assert_array_equal(h[4],[3,3,4])


def test_v40_mixed_support_schedule_is_protected_then_raw():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts.train_joint_action_adapter import _environment_safety_for_iteration
    assert [_environment_safety_for_iteration('protected',i) for i in range(3)] == [True,True,True]
    assert [_environment_safety_for_iteration('mixed',i) for i in range(3)] == [True,False,False]
    assert [_environment_safety_for_iteration('raw',i) for i in range(3)] == [False,False,False]
