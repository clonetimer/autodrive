from dataclasses import replace
import numpy as np
import pytest
import torch
from src.envs.native_world import NativeWorld
from src.envs.observation import build_observation, build_ego_motion
from src.perception.history import align_frame, BEVHistory
from src.agents.trainer import train, load_checkpoint
from src.utils.config import EnvConfig, TrainConfig
from src.perception.encoder import stack_observations, tensor_observation


def test_ego_motion_resolves_speed_alias_without_changing_geometry():
    env = NativeWorld(EnvConfig(observation='bev_ego', traffic_vehicles=0))
    env.reset(seed=3)
    ego = env.vehicles['agent0']
    ego.speed = 8
    slow = env.observe('agent0')
    ego.speed = 24
    fast = env.observe('agent0')
    np.testing.assert_array_equal(slow['bev'], fast['bev'])
    assert slow['motion'][0] != fast['motion'][0]
    assert env.observation_space.contains(fast)


@pytest.mark.parametrize('mode', ['bev', 'hybrid', 'bev_ego'])
def test_temporal_shape_masks_timestamp_and_reset_isolation(mode):
    env = NativeWorld(EnvConfig(observation=mode, bev_frames=4, traffic_vehicles=1))
    observations, _ = env.reset(seed=4)
    first = observations['agent0']
    assert first['bev'].shape == (20, 48, 48)
    np.testing.assert_array_equal(first['history'].reshape(4, 2)[:, 1], [1, 0, 0, 0])
    assert not first['bev'][5:].any()
    for i in range(3):
        observations, *_ = env.step({'agent0': [0, 0]})
        assert env.observation_space.contains(observations['agent0'])
    history = observations['agent0']['history'].reshape(4, 2)
    np.testing.assert_allclose(history[:, 0], [0, 1/3, 2/3, 1], atol=1e-6)
    np.testing.assert_array_equal(history[:, 1], np.ones(4))
    old_times = [frame.time_s for frame in env.histories['agent0'].frames]
    env.observe('agent0'); env.observe('agent0')
    assert old_times == [frame.time_s for frame in env.histories['agent0'].frames]
    reset_obs, _ = env.reset(seed=4)
    for key in first:
        np.testing.assert_array_equal(first[key], reset_obs['agent0'][key])


def test_ego_compensation_translates_static_pixels_not_dynamic_guess():
    # 65px: one row = 1m forward. Ego moves +4m, a static world object moves down four rows.
    image = np.zeros((5, 65, 65), np.uint8)
    image[3, 30, 32] = 255
    aligned = align_frame(image, (0, 0, 0), (4, 0, 0))
    assert aligned[3, 34, 32] == 255
    assert aligned.sum() == 255
    np.testing.assert_array_equal(align_frame(image, (0, 0, 0), (0, 0, 0)), image)


def test_capture_requires_monotonic_physical_time():
    env = NativeWorld(EnvConfig(traffic_vehicles=0))
    env.reset(seed=0)
    history = BEVHistory(4, 48, .1)
    history.capture(env.snapshot('agent0'), 0)
    with pytest.raises(ValueError):
        history.capture(env.snapshot('agent0'), 0)


@pytest.mark.parametrize('algorithm,agents', [('ppo', 1), ('mappo', 2), ('sac', 1)])
def test_temporal_bev_ego_actual_updates_and_cold_load(tmp_path, algorithm, agents):
    cfg = TrainConfig(algorithm=algorithm, total_steps=32, rollout_steps=16, epochs=1,
        batch_size=8, replay_size=64, learning_starts=8,
        env=EnvConfig(observation='bev_ego', bev_frames=4, bev_size=32, num_agents=agents,
                      traffic_vehicles=0, max_steps=8))
    result = train(cfg, tmp_path)
    assert result['encoder_changed_l1'] > 0
    model, loaded, _ = load_checkpoint(tmp_path / 'model.pt')
    env = NativeWorld(cfg.env)
    obs, _ = env.reset(seed=2)
    action = model.predict(obs['agent0'])
    assert action.shape == (2,) and np.isfinite(action).all()
    assert loaded.env.bev_frames == 4 and loaded.env.reward_version == 3
    if algorithm == 'mappo':
        batch = tensor_observation(stack_observations([obs['agent0']]))
        before = model.distribution(batch)[0].mean.detach().clone()
        with torch.no_grad():
            for parameter in model.critic.parameters():
                parameter.add_(100)
        assert torch.equal(before, model.distribution(batch)[0].mean.detach())


def test_historical_model_has_legacy_reward_and_no_history():
    from pathlib import Path
    checkpoint = Path(__file__).resolve().parents[1] / 'experiments/release/ppo_state_seed7/model.pt'
    model, cfg, _ = load_checkpoint(checkpoint)
    assert cfg.env.reward_version == 2 and cfg.env.bev_frames == 1


@pytest.mark.parametrize('kwargs', [{'bev_frames': 0}, {'bev_frames': 9}, {'bev_frames': 1.5},
                                    {'observation': 'state', 'bev_frames': 4}, {'reward_version': 99}])
def test_invalid_observation_or_reward_version(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_temporal_observation_separates_relative_motion_at_same_current_geometry():
    from src.scenarios.traffic import Vehicle
    observed = []
    for leader_speed in (2, 10):
        world = NativeWorld(EnvConfig(observation='bev_ego', bev_frames=4, bev_size=64, traffic_vehicles=0, dt=.2))
        world.reset(seed=1)
        ego = world.vehicles['agent0']
        history = BEVHistory(4, 64, .2)
        for step in range(4):
            # Prescribed past trajectories ending at the same geometry; no future frames.
            x = ego.x + 25 - leader_speed * (3 - step) * .2
            world.vehicles['lead'] = Vehicle('lead', x, ego.y, 0, leader_speed)
            history.capture(world.snapshot('agent0'), step * .2)
        obs = build_observation(world.snapshot('agent0'), world.config, world.rng, history)
        observed.append(obs)
    np.testing.assert_array_equal(observed[0]['bev'][:5], observed[1]['bev'][:5])
    np.testing.assert_array_equal(observed[0]['motion'], observed[1]['motion'])
    assert not np.array_equal(observed[0]['bev'][5:], observed[1]['bev'][5:])
