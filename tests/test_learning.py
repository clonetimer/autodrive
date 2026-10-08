from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.distributions import Normal
from src.agents.policy import DrivingPolicy
from src.agents.rollout import generalized_advantage
from src.agents.sac_agent import ReplayBuffer
from src.agents.trainer import train, load_checkpoint, save_checkpoint
from src.envs.native_world import NativeWorld
from src.utils.config import EnvConfig, TrainConfig
from src.end_to_end.agent import EndToEndAgent
from src.end_to_end.system import run_episode


def test_gae_termination_truncation_difference():
    reward = np.array([[1, 1]], np.float32)
    value = np.array([[0.5, 0.5]], np.float32)
    next_value = np.array([[2, 2]], np.float32)
    advantage, returns = generalized_advantage(reward, value, next_value, [[True, False]], [[False, True]], [[True, True]], gamma=0.9)
    np.testing.assert_allclose(returns, [[1, 2.8]])
    np.testing.assert_allclose(advantage, [[0.5, 2.3]])


def test_gae_does_not_cross_episode_boundaries():
    advantage, _ = generalized_advantage([[1], [100]], [[0], [0]], [[0], [0]],
        [[False], [False]], [[True], [True]], [[True], [True]], gamma=1, gae_lambda=1)
    np.testing.assert_array_equal(advantage, [[1], [100]])


def test_gae_masks_dead_agents():
    advantage, _ = generalized_advantage([[1, 5], [100, 7]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
        [[True, False], [True, True]], [[False, False], [False, False]], [[True, True], [False, True]])
    assert advantage[1, 0] == 0 and advantage[0, 0] == 1


def test_squashed_logprob_finite_at_extremes():
    distribution = Normal(torch.zeros(2, 2), torch.ones(2, 2))
    latent = torch.tensor([[50., -50.], [0., 0.]])
    values = DrivingPolicy.log_probability(distribution, latent)
    assert torch.isfinite(values).all()
    assert values[1].item() == pytest.approx(-np.log(2 * np.pi), abs=1e-6)


@pytest.mark.parametrize('algorithm,mode,n_agents', [('ppo', 'state', 1), ('mappo', 'state', 2), ('ppo', 'bev', 1), ('mappo', 'hybrid', 2), ('sac', 'state', 1)])
def test_actual_training_updates_and_reload(tmp_path, algorithm, mode, n_agents):
    cfg = TrainConfig(algorithm=algorithm, total_steps=64, rollout_steps=32, epochs=2, batch_size=16,
        learning_starts=16, replay_size=128, checkpoint_every=32,
        env=EnvConfig(observation=mode, num_agents=n_agents, traffic_vehicles=1, max_steps=30, road_length=70))
    output = tmp_path / algorithm / mode
    result = train(cfg, output)
    assert result['parameters_changed_l1'] > 0
    assert result['encoder_changed_l1'] > 0
    assert result['world_steps_total'] == 64
    assert (output / 'model.pt').exists()
    assert list((output / 'tensorboard').glob('events.out.tfevents.*'))
    policy = EndToEndAgent(output / 'model.pt')
    env = NativeWorld(cfg.env)
    obs, _ = env.reset(seed=0)
    action1 = policy.predict(obs['agent0'])
    second = EndToEndAgent(output / 'model.pt')
    np.testing.assert_array_equal(action1, second.predict(obs['agent0']))
    assert action1.shape == (2,) and np.isfinite(action1).all()
    path = output / 'rollout.json'
    trajectory = run_episode(output / 'model.pt', output=path)
    assert path.exists() and trajectory['trajectory']


def test_ppo_optimizer_resume_counter(tmp_path):
    cfg = TrainConfig(total_steps=32, rollout_steps=16, epochs=1, batch_size=8,
                      env=EnvConfig(traffic_vehicles=0, max_steps=10))
    train(cfg, tmp_path / 'first')
    result = train(replace(cfg, total_steps=64), tmp_path / 'resume', tmp_path / 'first/model.pt')
    assert result['world_steps_this_run'] == 32
    assert result['world_steps_total'] == 64


def test_replay_ring_and_uint8():
    example = {'bev': np.zeros((5, 32, 32), dtype=np.uint8)}
    buffer = ReplayBuffer(3, example, seed=1)
    for step in range(5):
        buffer.add(example, [step, 0], float(step), example, step == 4)
    assert buffer.size == 3 and buffer.position == 2
    assert buffer.obs['bev'].dtype == np.uint8
    obs, actions, rewards, next_obs, terminated = buffer.sample(5)
    assert obs['bev'].shape == (5, 5, 32, 32)
    assert actions.shape == (5, 2)
    assert (rewards >= 2).all()


def test_cpu_repeat_training_reproduces_actions(tmp_path):
    cfg = TrainConfig(total_steps=48, rollout_steps=24, epochs=1, batch_size=12, seed=5,
                      env=EnvConfig(traffic_vehicles=0, max_steps=20, seed=5))
    train(cfg, tmp_path / 'a')
    train(cfg, tmp_path / 'b')
    a, _, _ = load_checkpoint(tmp_path / 'a/model.pt')
    b, _, _ = load_checkpoint(tmp_path / 'b/model.pt')
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_measured_curriculum_promotes_only_joint_success_and_resume(tmp_path):
    from src.agents.trainer import PPOTrainer
    cfg = TrainConfig(total_steps=64, rollout_steps=16, epochs=1, batch_size=8,
        curriculum=True, curriculum_min_episodes=1, curriculum_success_threshold=1,
        curriculum_max_collision_rate=0, env=EnvConfig(num_agents=2, traffic_vehicles=3, max_steps=4))
    trainer = PPOTrainer(cfg, tmp_path / 'curriculum')
    assert trainer.env.config.traffic_vehicles == 0
    trainer.joint_outcomes = [(True, False), (False, True)]
    assert not trainer._record_joint_outcome()
    assert trainer.curriculum.stage == 0
    trainer.curriculum.history.clear()
    trainer.joint_outcomes = [(True, False), (True, False)]
    assert trainer._record_joint_outcome()
    assert trainer.curriculum.stage == 1 and trainer.env.config.traffic_vehicles == 5
    path = tmp_path / 'stage.pt'
    save_checkpoint(path, trainer.model, cfg, trainer.optimizer, 10, trainer._checkpoint_extra())
    trainer.writer.close()
    trainer.env.close()
    restored = PPOTrainer(cfg, tmp_path / 'restored', path)
    assert restored.curriculum.stage == 1 and restored.env.config.traffic_vehicles == 5
    restored.train()
    assert (tmp_path / 'restored/curriculum.csv').exists()


def test_public_ppo_agent_and_mappo_trainer(tmp_path):
    from src.agents.ppo_agent import PPOAgent
    from src.multi_agent.trainer import MAPPOTrainer
    cfg = TrainConfig(algorithm='mappo', total_steps=16, rollout_steps=16, epochs=1, batch_size=8,
                      env=EnvConfig(num_agents=2, traffic_vehicles=0, max_steps=8))
    trainer = MAPPOTrainer(cfg, tmp_path / 'mappo')
    trainer.train()
    agent = PPOAgent(tmp_path / 'mappo/model.pt')
    agent.reset()
    world = NativeWorld(cfg.env)
    obs, _ = world.reset(seed=1)
    assert agent.predict(obs['agent0']).shape == (2,)
    world.close()
    with pytest.raises(ValueError):
        MAPPOTrainer(replace(cfg, algorithm='ppo'), tmp_path / 'bad')
