"""Real MetaDrive smoke for the v3.1 learned steering adapter."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from src.agents.trainer import load_checkpoint
from src.end_to_end.agent import EndToEndAgent
from src.envs.metadrive_wrapper import MetaDriveParallelEnv, prerequisite_status
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent

READY = prerequisite_status()['ready']


@pytest.mark.metadrive
@pytest.mark.skipif(not READY, reason='Real MetaDrive/Panda3D/Gymnasium/assets not available')
def test_real_metadrive_learned_adapter_runs_without_geometric_route_assist():
    model_path = 'models/metadrive_ppo_state_capability_distilled_v28.pt'
    adapter_path = 'models/metadrive_learned_route_adapter_v31.pt'
    _, train_cfg, _ = load_checkpoint(model_path)
    cfg = replace(train_cfg.env, max_steps=12, metadrive_map='SCS', metadrive_traffic_density=.05,
                  target_speed=8.0, route_assist=False, safety_supervisor=False)
    env = MetaDriveParallelEnv(cfg)
    base = EndToEndAgent(model_path)
    adapted = LearnedRouteAdapterAgent(model_path, adapter_path)
    try:
        obs, _ = env.reset(seed=31050)  # training-domain diagnostic, never a release holdout.
        state = obs['agent0']
        raw_action = base.predict(state)
        adapter_action = adapted.predict(state)
        assert adapter_action.shape == (2,)
        assert np.isfinite(adapter_action).all()
        assert adapter_action[1] == pytest.approx(raw_action[1])
        assert adapted.last_diagnostics['learned_adapter_blend'] >= 0.0
        _, _, _, _, infos = env.step({'agent0': adapter_action})
        info = infos['agent0']
        assert info['route_assist'] is False
        assert info['route_assist_blend'] == pytest.approx(0.0)
        np.testing.assert_allclose(info['proposed_action'], adapter_action)
    finally:
        env.close()
