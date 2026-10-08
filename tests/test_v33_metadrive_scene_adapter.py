"""Real MetaDrive runtime contract for the v3.3 temporal adapter architecture."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from src.agents.trainer import load_checkpoint
from src.envs.metadrive_wrapper import MetaDriveParallelEnv, prerequisite_status
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.navigation.scene_aware_adapter import (
    SceneAwareTemporalAdapter, SceneAwareTemporalAdapterConfig, save_scene_aware_adapter)

READY = prerequisite_status()['ready']


@pytest.mark.metadrive
@pytest.mark.skipif(not READY, reason='Real MetaDrive/Panda3D/Gymnasium/assets not available')
def test_real_metadrive_v33_temporal_runtime_uses_no_rule_controller(tmp_path):
    model_path = 'models/metadrive_ppo_state_capability_distilled_v28.pt'
    _, train_cfg, _ = load_checkpoint(model_path)
    state_dim = 20 + 7 * train_cfg.env.neighbors
    model = SceneAwareTemporalAdapter(SceneAwareTemporalAdapterConfig(
        state_dim, hidden_size=64, history_len=4, neighbor_heads=4))
    adapter_path = tmp_path / 'identity_v33.pt'
    save_scene_aware_adapter(adapter_path, model, base_model=model_path, training={'runtime_smoke': True})

    cfg = replace(
        train_cfg.env, max_steps=12, metadrive_map='SCS',
        metadrive_traffic_density=.05, target_speed=8.0,
        route_assist=False, safety_supervisor=False)
    env = MetaDriveParallelEnv(cfg)
    agent = LearnedRouteAdapterAgent(model_path, adapter_path)
    try:
        obs, _ = env.reset(seed=37050)  # scene_train diagnostic domain, never a release holdout.
        agent.reset()
        for expected_history in (1, 2):
            action = agent.predict(obs['agent0'])
            assert action.shape == (2,) and np.isfinite(action).all()
            assert agent.last_diagnostics['learned_adapter_history_frames'] == pytest.approx(expected_history)
            nxt, _, _, _, infos = env.step({'agent0': action})
            info = infos['agent0']
            assert info['route_assist'] is False
            assert info['safety_supervisor'] is False
            np.testing.assert_allclose(info['proposed_action'], action)
            obs = {name: nxt[name] for name in env.agents}
    finally:
        env.close()
