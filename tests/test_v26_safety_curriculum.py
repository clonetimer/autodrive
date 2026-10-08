import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from src.agents.distillation import actor_actions
from src.agents.policy import DrivingPolicy
from src.evaluation.safety_suite import standard_safety_suite
from src.safety.curriculum import load_safety_curriculum
from src.safety.distillation import distill_safety_actor
from src.scenarios.generator import SPLITS, split_seeds
from src.utils.config import EnvConfig


def test_release5_is_disjoint_from_training_and_prior_release_domains():
    release5 = set(split_seeds('release5', 200))
    assert release5 == set(range(9000, 9200))
    for name, (lo, hi) in SPLITS.items():
        if name != 'release5':
            assert release5.isdisjoint(range(lo, hi + 1))


def test_default_safety_curriculum_is_progressive_and_applies_to_metadrive():
    stages = load_safety_curriculum()
    assert [s.name for s in stages] == ['straight_moderate', 'straight_dense', 'curve_dense']
    assert stages[1].traffic_density > stages[0].traffic_density
    base = EnvConfig(backend='metadrive', observation='state', reward_source='custom', reward_version=4)
    changed = stages[2].apply(base)
    assert changed.metadrive_map == 'SCS' and changed.target_speed == 13.0
    assert base.metadrive_map == 'S'  # dataclass replacement, no hidden mutation


def test_safety_suite_uses_unique_scenarios_and_contains_unseen_structure():
    suite = standard_safety_suite('release5', 4)
    seeds = [seed for stage in suite for seed in stage.seeds]
    assert len(seeds) == len(set(seeds)) == 16
    assert suite[-1].name == 'mixed_unseen' and suite[-1].metadrive_map == 'SCXTO'


def test_longitudinal_only_distillation_preserves_encoder_and_steering_row_exactly():
    torch.manual_seed(26)
    rng = np.random.default_rng(26)
    model = DrivingPolicy('state', neighbors=1, hidden_size=24)
    states = rng.normal(size=(128, 27)).astype(np.float32)
    with torch.no_grad():
        proposed = actor_actions(model, torch.as_tensor(states)).cpu().numpy()
    targets = proposed.copy()
    intervened = np.zeros(128, dtype=bool)
    intervened[::2] = True
    targets[intervened, 1] = np.clip(targets[intervened, 1] - .65, -1, 1)
    severity = np.where(intervened, .8, 0).astype(np.float32)
    encoder_before = copy.deepcopy(model.encoder.state_dict())
    steer_weight_before = model.actor.weight[0].detach().clone()
    steer_bias_before = model.actor.bias[0].detach().clone()
    longitudinal_before = model.actor.weight[1].detach().clone()
    critic_before = copy.deepcopy(model.critic.state_dict())
    log_std_before = model.log_std.detach().clone()

    result = distill_safety_actor(
        model, states, targets, intervened, severity, epochs=12, batch_size=32,
        learning_rate=8e-4, intervention_weight=2.0, steering_weight=.35,
        train_encoder=False, longitudinal_only=True, seed=26)

    assert result.final_intervention_mse < result.initial_intervention_mse
    assert result.encoder_changed_l1 == 0.0
    assert result.steering_actor_changed_l1 == 0.0
    assert result.longitudinal_actor_changed_l1 > 0
    for name, value in encoder_before.items():
        torch.testing.assert_close(model.encoder.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model.actor.weight[0], steer_weight_before, rtol=0, atol=0)
    torch.testing.assert_close(model.actor.bias[0], steer_bias_before, rtol=0, atol=0)
    assert not torch.equal(model.actor.weight[1], longitudinal_before)
    for name, value in critic_before.items():
        torch.testing.assert_close(model.critic.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model.log_std, log_std_before, rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_longitudinal_only_rejects_trainable_encoder():
    model = DrivingPolicy('state', neighbors=1, hidden_size=16)
    states = np.zeros((4, 27), np.float32)
    targets = np.zeros((4, 2), np.float32)
    intervened = np.array([1, 0, 0, 0], bool)
    severity = np.array([.5, 0, 0, 0], np.float32)
    with pytest.raises(ValueError):
        distill_safety_actor(model, states, targets, intervened, severity,
                             train_encoder=True, longitudinal_only=True)


def test_lane_teacher_is_inactive_on_centered_straight_and_bounded_on_drift():
    from src.safety.lane_teacher import lane_teacher_action
    state = np.zeros(62, dtype=np.float32)
    state[2] = 1.0
    state[12] = 8.0 / 60.0
    action = np.array([0.15, 0.2], dtype=np.float32)
    straight = lane_teacher_action(action, state)
    assert not straight.intervened
    assert straight.target_steer == pytest.approx(float(action[0]))

    drift = state.copy()
    drift[3] = 0.70
    drift[1] = 0.35
    drift[13] = 2.5 / 20.0
    correction = lane_teacher_action(action, drift, max_blend=.8)
    assert correction.intervened and 0 < correction.severity <= 1
    assert -1 <= correction.target_steer <= 1
    assert abs(correction.target_steer - action[0]) > 1e-3


def test_lane_teacher_rejects_invalid_inputs():
    from src.safety.lane_teacher import lane_teacher_action
    with pytest.raises(ValueError):
        lane_teacher_action([0, 0], np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        lane_teacher_action([0, np.nan], np.zeros(62, dtype=np.float32))
