from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from src.agents.policy import DrivingPolicy
from src.envs.native_world import NativeWorld
from src.envs.observation import state_dimension
from src.perception.encoder import tensor_observation, stack_observations
from src.perception.bev_generator import local_to_pixels, rasterize
from src.utils.config import EnvConfig


def test_shared_world_action_coupling():
    cfg = EnvConfig(num_agents=2, traffic_vehicles=0)
    first, second = NativeWorld(cfg), NativeWorld(cfg)
    for env in (first, second):
        env.reset(seed=5)
        env.vehicles['agent0'].x = 30
        env.vehicles['agent1'].y = env.vehicles['agent0'].y
    for _ in range(10):
        one = first.step({'agent0': [0, 1], 'agent1': [0, 0]})
        two = second.step({'agent0': [0, -1], 'agent1': [0, 0]})
    assert not np.array_equal(one[0]['agent1'][20:], two[0]['agent1'][20:])
    assert len(first.vehicles) == 2  # not two independent simulator copies


def test_two_agents_physically_collide():
    env = NativeWorld(EnvConfig(num_agents=2, traffic_vehicles=0))
    env.reset(seed=0)
    env.vehicles['agent1'].y = env.vehicles['agent0'].y
    env.vehicles['agent1'].x = env.vehicles['agent0'].x
    obs, _, terms, _, infos = env.step({name: [0, 0] for name in env.agents})
    assert all(terms.values())
    assert all(info['collision'] for info in infos.values())
    assert len(obs) == 2 and len(env.agents) == 0


def test_individual_agent_lifetime():
    env = NativeWorld(EnvConfig(num_agents=2, traffic_vehicles=0))
    env.reset(seed=3)
    env.vehicles['agent0'].y = 50
    obs, _, terms, truncs, _ = env.step({name: [0, 0] for name in env.agents})
    assert terms['agent0'] and not terms['agent1']
    assert env.agents == ['agent1']
    obs, *_ = env.step({'agent1': [0, 0]})
    assert set(obs) == {'agent1'}
    assert env.global_state().shape == (2 * (state_dimension() + 1),)


def test_terminal_central_state_keeps_time_limit_state():
    env = NativeWorld(EnvConfig(num_agents=2, traffic_vehicles=0, max_steps=1))
    env.reset(seed=2)
    env.step({name: [0, 0] for name in env.agents})
    assert env.transition_state.sum() != 0
    assert env.global_state().sum() == 0


@pytest.mark.parametrize('mode', ['state', 'bev', 'hybrid'])
def test_policy_shape_and_gradient(mode):
    env = NativeWorld(EnvConfig(observation=mode, traffic_vehicles=1))
    obs, _ = env.reset(seed=3)
    policy = DrivingPolicy(mode)
    tensors = tensor_observation(stack_observations([obs['agent0']] * 2))
    action, latent, logp, value = policy.sample(tensors)
    assert action.shape == (2, 2)
    assert torch.isfinite(action).all() and (action.abs() <= 1).all()
    loss = -logp.mean() + value.square().mean()
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in policy.encoder.parameters())


def test_central_state_changes_critic_not_actor():
    torch.manual_seed(6)
    policy = DrivingPolicy('state', central_dim=20)
    obs = torch.zeros(3, state_dimension())
    zero = torch.zeros(3, 20)
    one = torch.ones(3, 20)
    a, _, _, v1 = policy.sample(obs, zero, deterministic=True)
    b, _, _, v2 = policy.sample(obs, one, deterministic=True)
    assert torch.equal(a, b)
    assert not torch.equal(v1, v2)


def test_bev_translation_invariance(world):
    original = world.snapshot('agent0')
    first = rasterize(original)
    shifted = copy.deepcopy(original)
    # ego also occurs in vehicles by reference; deepcopy preserves aliasing.
    shifted_ids = set()
    for car in [shifted.ego] + shifted.vehicles:
        if id(car) not in shifted_ids:
            car.x += 100
            car.y -= 20
            shifted_ids.add(id(car))
    shifted.boundaries = [line + [100, -20] for line in shifted.boundaries]
    shifted.route = shifted.route + [100, -20]
    second = rasterize(shifted)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 48, 48) and first.dtype == np.uint8
    assert first[0].sum() > 0 and first[4].sum() > 0


def test_forward_maps_up():
    pixels = local_to_pixels(np.array([[0, 0], [10, 0], [0, 5]]), 48)
    assert pixels[1, 1] < pixels[0, 1]
    assert pixels[2, 0] < pixels[0, 0]


def test_neighbor_padding_and_noise_mask():
    env = NativeWorld(EnvConfig(traffic_vehicles=0, observation_noise=0.02))
    obs, _ = env.reset(seed=8)
    assert np.all(obs['agent0'][20:] == 0)
