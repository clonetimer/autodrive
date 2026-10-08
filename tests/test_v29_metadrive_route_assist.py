"""Small real-MetaDrive contract test for the transparent v2.9 hybrid stack."""
from __future__ import annotations

import numpy as np
import pytest

from src.envs.metadrive_wrapper import MetaDriveParallelEnv, prerequisite_status
from src.utils.config import EnvConfig

READY = prerequisite_status()['ready']


@pytest.mark.metadrive
@pytest.mark.skipif(not READY, reason='Real MetaDrive/Panda3D/Gymnasium/assets not available')
def test_real_metadrive_route_assist_and_longitudinal_supervisor_are_separable():
    cfg = EnvConfig(
        backend='metadrive', observation='state', num_agents=1, max_steps=15,
        metadrive_map='SCS', metadrive_traffic_density=0.05, target_speed=8.0,
        route_assist=True, route_assist_blend=1.0,
        safety_supervisor=True, safety_ttc_threshold_s=1.8, safety_risk_threshold=0.68,
    )
    env = MetaDriveParallelEnv(cfg)
    try:
        obs, _ = env.reset(seed=26000)
        assert obs['agent0'].ndim == 1
        proposed = np.array([0.85, 0.9], dtype=np.float32)
        _, _, _, _, infos = env.step({'agent0': proposed})
        info = infos['agent0']
        assert info['route_assist'] is True
        assert info['safety_supervisor'] is True
        np.testing.assert_allclose(info['proposed_action'], proposed)
        # Lateral route assist is forbidden from changing longitudinal command.
        assert info['route_assisted_action'][1] == pytest.approx(proposed[1])
        # Longitudinal safety is forbidden from changing route-assisted steering.
        assert info['applied_action'][0] == pytest.approx(info['route_assisted_action'][0])
        assert info['route_assist_steering_correction'] >= 0.0
        assert info['safety_severity'] >= 0.0
    finally:
        env.close()
