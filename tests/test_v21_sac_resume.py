from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
import torch
from src.agents.sac_agent import ReplayBuffer, SACTrainer
from src.agents.trainer import train
from src.utils.config import TrainConfig, EnvConfig


def config():
    return TrainConfig(algorithm='sac', total_steps=48, learning_starts=8, batch_size=8,
        rollout_steps=16, checkpoint_every=32, replay_size=64,
        env=EnvConfig(traffic_vehicles=0, max_steps=12))


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('dictionary', [False, True])
def test_replay_complete_ring_and_sampling_rng_roundtrip(dictionary, tmp_path):
    example = {'bev': np.zeros((5, 32, 32), np.uint8), 'motion': np.zeros(20, np.float32)} if dictionary else np.zeros(6, np.float32)
    a = ReplayBuffer(7, example, 3)
    for i in range(13):
        a.add(example, np.array([i / 13, 0], np.float32), i, example, i == 12)
    path = tmp_path / 'replay.pt'
    torch.save(a.state_dict(), path)
    b = ReplayBuffer(7, example, 99)
    b.load_state_dict(torch.load(path, weights_only=True))
    assert b.position == a.position and b.size == a.size
    assert_nested_equal(a.sample(12), b.sample(12))
    a.add(example, [0, 0], 100, example, False)
    b.add(example, [0, 0], 100, example, False)
    assert_nested_equal(a.state_dict(), b.state_dict())


def test_full_resume_restores_learning_state_and_identical_next_update(tmp_path):
    cfg = config()
    a = SACTrainer(cfg, tmp_path / 'a')
    try:
        for i in range(20):
            action = a.rng.uniform(-1, 1, 2).astype(np.float32)
            no, r, term, trunc, _ = a.env.step({'agent0': action})
            a.buffer.add(a.obs['agent0'], action, r['agent0'], no['agent0'], term['agent0'])
            a.obs = a.env.reset()[0] if not a.env.agents else no
        a.steps, a.updates, a.episodes = 20, 1, 1
        a.update()
        path = tmp_path / 'state.pt'
        a.save_training_checkpoint(path)
        expected_metrics = a.update()
        expected = {key: deepcopy(getattr(a, key).state_dict()) for key in
                    ['actor', 'critic', 'target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer']}
        expected_alpha = a.log_alpha.detach().clone()
        b = SACTrainer(cfg, tmp_path / 'b', resume=path)
        try:
            assert (b.steps, b.updates, b.episodes) == (20, 1, 1)
            assert b.env.steps == 0  # honest new-episode semantics, not physical continuation
            assert b.update() == pytest.approx(expected_metrics, abs=0, rel=0)
            for key, value in expected.items():
                assert_nested_equal(value, getattr(b, key).state_dict())
            assert torch.equal(expected_alpha, b.log_alpha.detach())
        finally:
            b.env.close(); b.writer.close()
    finally:
        a.env.close(); a.writer.close()


def test_train_cli_semantics_absolute_counter_and_full_artifact(tmp_path):
    cfg = config()
    first = train(cfg, tmp_path / 'first')
    assert (tmp_path / 'first/training_state.pt').exists()
    result = train(replace(cfg, total_steps=80), tmp_path / 'second', tmp_path / 'first/training_state.pt')
    assert result['world_steps_total'] == 80 and result['world_steps_this_run'] == 32
    assert result['resumed'] and result['updates_this_run'] == 32
    assert not list((tmp_path / 'second').rglob('*.tmp'))
    with pytest.raises(ValueError, match='actor-only'):
        SACTrainer(replace(cfg, total_steps=80), tmp_path / 'bad', tmp_path / 'first/model.pt')
    with pytest.raises(ValueError, match='config mismatch'):
        SACTrainer(replace(cfg, total_steps=80, env=replace(cfg.env, reward_version=2)),
                   tmp_path / 'bad2', tmp_path / 'first/training_state.pt')


def test_pixel_replay_memory_guard_and_shape_validation():
    with pytest.raises(ValueError, match='Reduce replay_size'):
        ReplayBuffer(50000, {'bev': np.zeros((20, 48, 48), np.uint8)}, 0)
    b = ReplayBuffer(4, np.zeros(3, np.float32), 0)
    state = b.state_dict()
    state['position'] = 2
    with pytest.raises(ValueError, match='partial replay'):
        b.load_state_dict(state)
