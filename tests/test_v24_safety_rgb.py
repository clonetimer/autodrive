"""v2.4 safety reward and raw-RGB policy contracts."""
from __future__ import annotations
import numpy as np
import pytest
import torch

from src.agents.policy import DrivingPolicy
from src.envs.native_world import NativeWorld
from src.envs.observation import build_ego_motion, observation_space
from src.envs.reward import reward_terms, traffic_risk
from src.perception.encoder import tensor_observation
from src.scenarios.traffic import Vehicle
from src.utils.config import EnvConfig


def emergency_snapshot(acceleration: float):
    env = NativeWorld(EnvConfig(traffic_vehicles=0, target_speed=15, initial_speed=15, reward_version=4))
    env.reset(seed=3)
    ego = env.vehicles['agent0']
    ego.acceleration = acceleration
    env.vehicles['leader'] = Vehicle('leader', ego.x + 15, ego.y, ego.yaw, 0, 0)
    return env.snapshot('agent0')


def test_v4_penalizes_accelerating_into_short_ttc_more_than_braking():
    accelerating = emergency_snapshot(3.0)
    braking = emergency_snapshot(-4.0)
    risk = traffic_risk(accelerating)
    assert risk.ttc_s is not None and risk.ttc_s < 1.0
    a = reward_terms(accelerating, 1.0, 0.0, .1, 'safe', 4, risk=risk)
    b_risk = traffic_risk(braking)
    b = reward_terms(braking, 1.0, 0.0, .1, 'safe', 4, risk=b_risk)
    assert a['ttc'] < 0 and a['unsafe_accel'] < 0
    assert b['emergency_brake'] > 0
    assert sum(b.values()) > sum(a.values())


def test_v4_has_no_emergency_brake_credit_on_empty_road():
    env = NativeWorld(EnvConfig(traffic_vehicles=0, target_speed=15, initial_speed=15, reward_version=4))
    env.reset(seed=4)
    env.vehicles['agent0'].acceleration = -4.0
    terms = reward_terms(env.snapshot('agent0'), 1.0, 0.0, .1, 'balanced', 4)
    assert terms['emergency_brake'] == pytest.approx(0.0)
    assert terms['ttc'] == pytest.approx(0.0)


def test_rgb_config_is_explicit_metadrive_only():
    with pytest.raises(ValueError):
        EnvConfig(observation='rgb_ego')
    cfg = EnvConfig(backend='metadrive', observation='rgb_ego', rgb_width=80, rgb_height=40)
    space = observation_space('rgb_ego', cfg.neighbors, cfg.bev_size, 1, cfg.rgb_width, cfg.rgb_height)
    sample = {'rgb': np.zeros((3, 40, 80), np.uint8), 'motion': np.zeros(20, np.float32)}
    assert space.contains(sample)


def test_rgb_ego_policy_has_trainable_pixel_encoder():
    torch.manual_seed(2)
    policy = DrivingPolicy('rgb_ego', hidden_size=32)
    batch = {'rgb': np.random.default_rng(2).integers(0, 256, size=(4, 3, 40, 80), dtype=np.uint8),
             'motion': np.zeros((4, 20), np.float32)}
    obs = tensor_observation(batch)
    action, _, logp, value = policy.sample(obs)
    loss = -logp.mean() + value.square().mean()
    loss.backward()
    assert action.shape == (4, 2)
    rgb_grads = [p.grad for name, p in policy.encoder.named_parameters() if name.startswith('rgb.')]
    assert rgb_grads and any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in rgb_grads)


def test_ego_motion_keeps_navigation_without_privileged_neighbors():
    env = NativeWorld(EnvConfig(traffic_vehicles=2))
    env.reset(seed=9)
    motion = build_ego_motion(env.snapshot('agent0'))
    assert motion.shape == (20,)
    assert np.isfinite(motion).all()


def test_metadrive_custom_reward_is_actually_selected_when_runtime_available():
    pytest.importorskip('metadrive')
    from src.envs.metadrive_wrapper import MetaDriveParallelEnv
    cfg = EnvConfig(backend='metadrive', observation='state', reward_source='custom', reward_version=4,
                    reward='safe', num_agents=1, max_steps=20, substeps=5,
                    metadrive_traffic_density=0.1, seed=321)
    env = MetaDriveParallelEnv(cfg)
    try:
        obs, _ = env.reset(seed=321)
        assert obs['agent0'].shape[0] > 20
        _, rewards, _, _, infos = env.step({'agent0': np.array([0.0, 0.2], np.float32)})
        info = infos['agent0']
        assert info['reward_source'] == 'custom'
        assert rewards['agent0'] == pytest.approx(info['custom_reward'])
        assert 'official_reward' in info and 'ttc' in info['reward_terms']
        assert np.isfinite(rewards['agent0'])
    finally:
        env.close()


def test_final_seed_split_is_disjoint_from_development_splits():
    from src.scenarios.generator import SPLITS, split_seeds
    final = set(split_seeds('final', 200))
    for name in ('train', 'validation', 'test'):
        lo, hi = SPLITS[name]
        assert final.isdisjoint(range(lo, hi + 1))


def test_real_metadrive_snapshot_contains_exactly_one_learning_ego_and_no_zero_gap_phantom():
    """Regression: engine object UUID != env.agents key must not duplicate ego."""
    pytest.importorskip('metadrive')
    from src.envs.metadrive_wrapper import MetaDriveParallelEnv
    cfg = EnvConfig(backend='metadrive', observation='state', reward_source='custom',
                    reward_version=4, num_agents=1, max_steps=20,
                    metadrive_traffic_density=0.0, seed=123)
    env = MetaDriveParallelEnv(cfg)
    try:
        env.reset(seed=123)
        snap = env._last_snapshots['agent0']
        assert sum(v.id == 'agent0' for v in snap.vehicles) == 1
        risk = traffic_risk(snap)
        assert risk.front_id is None
        assert risk.gap_m is None
        assert risk.front_risk == 0.0
        assert risk.safe_target_mps == pytest.approx(cfg.target_speed)
    finally:
        env.close()
