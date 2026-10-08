import copy
import numpy as np
import pytest
import torch
from src.agents.policy import DrivingPolicy
from src.safety.distillation import distill_safety_actor
from src.scenarios.generator import SPLITS, split_seeds


def test_release4_and_safety_train_splits_are_disjoint_from_prior_release_splits():
    release4 = set(split_seeds('release4', 200))
    safety_train = set(split_seeds('safety_train', 200))
    assert release4.isdisjoint(safety_train)
    for name in ('train', 'validation', 'test', 'final', 'release', 'release2', 'release3'):
        lo, hi = SPLITS[name]
        assert release4.isdisjoint(range(lo, hi + 1))
        assert safety_train.isdisjoint(range(lo, hi + 1))


def test_safety_distillation_reduces_intervention_target_error_and_keeps_critic_frozen():
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    policy = DrivingPolicy('state', neighbors=1, hidden_size=32)
    states = rng.normal(size=(96, 27)).astype(np.float32)
    with torch.no_grad():
        base = policy.predict(states[0])  # shape smoke
    # Build targets from the current policy, then impose supervisor-like
    # longitudinal braking on half the samples while preserving steering.
    from src.agents.distillation import actor_actions
    obs = torch.as_tensor(states)
    with torch.no_grad():
        proposed = actor_actions(policy, obs).cpu().numpy()
    intervened = np.zeros(96, dtype=bool)
    intervened[::2] = True
    severity = np.zeros(96, dtype=np.float32)
    severity[intervened] = .8
    targets = proposed.copy()
    targets[intervened, 1] = np.clip(targets[intervened, 1] - .7, -1, 1)
    critic_before = copy.deepcopy(policy.critic.state_dict())
    log_std_before = policy.log_std.detach().clone()
    result = distill_safety_actor(policy, states, targets, intervened, severity,
                                  epochs=12, batch_size=32, learning_rate=8e-4, seed=5)
    assert result.final_mse < result.initial_mse
    assert result.final_intervention_mse < result.initial_intervention_mse
    assert result.actor_encoder_changed_l1 > 0
    for name, value in critic_before.items():
        torch.testing.assert_close(policy.critic.state_dict()[name], value)
    torch.testing.assert_close(policy.log_std, log_std_before)


def test_safety_distillation_rejects_invalid_severity():
    policy = DrivingPolicy('state', neighbors=1, hidden_size=16)
    states = np.zeros((4, 27), np.float32)
    targets = np.zeros((4, 2), np.float32)
    with pytest.raises(ValueError):
        distill_safety_actor(policy, states, targets, np.zeros(4, bool), np.array([0, 0, 0, 1.2], np.float32))


def test_safety_distillation_does_not_drift_when_supervisor_never_intervenes():
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    policy = DrivingPolicy('state', neighbors=1, hidden_size=16)
    states = rng.normal(size=(24, 27)).astype(np.float32)
    from src.agents.distillation import actor_actions
    with torch.no_grad():
        targets = actor_actions(policy, torch.as_tensor(states)).cpu().numpy()
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    result = distill_safety_actor(policy, states, targets, np.zeros(24, bool), np.zeros(24, np.float32), epochs=4)
    assert result.batches == 0 and result.actor_encoder_changed_l1 == 0.0
    for name, value in before.items():
        torch.testing.assert_close(policy.state_dict()[name], value)
