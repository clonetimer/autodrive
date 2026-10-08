import copy
import numpy as np
import pytest
import torch

from src.agents.distillation import actor_actions
from src.agents.policy import DrivingPolicy
from src.evaluation.model_selection import select_safety_candidate
from src.evaluation.safety_suite import standard_safety_suite
from src.safety.distillation import distill_safety_actor
from src.scenarios.generator import SPLITS, split_seeds


def test_v27_fresh_domains_are_disjoint_from_all_prior_domains():
    for name in ('expert_train', 'validation2', 'release6'):
        lo, hi = SPLITS[name]
        values = set(split_seeds(name, hi - lo + 1))
        assert values == set(range(lo, hi + 1))
        for other, (olo, ohi) in SPLITS.items():
            if other != name:
                assert values.isdisjoint(range(olo, ohi + 1))


def test_standard_suite_accepts_validation2_and_release6():
    validation = standard_safety_suite('validation2', 3)
    release = standard_safety_suite('release6', 3)
    assert [s.seeds[0] for s in validation] == [11000, 11003, 11006, 11009]
    assert [s.seeds[0] for s in release] == [12000, 12003, 12006, 12009]


def test_steering_only_distillation_preserves_encoder_and_longitudinal_row_exactly():
    torch.manual_seed(27)
    rng = np.random.default_rng(27)
    model = DrivingPolicy('state', neighbors=1, hidden_size=24)
    states = rng.normal(size=(128, 27)).astype(np.float32)
    with torch.no_grad():
        proposed = actor_actions(model, torch.as_tensor(states)).cpu().numpy()
    targets = proposed.copy()
    intervened = np.zeros(128, dtype=bool)
    intervened[::2] = True
    targets[intervened, 0] = np.clip(targets[intervened, 0] + .65, -1, 1)
    severity = np.where(intervened, .8, 0).astype(np.float32)
    encoder_before = copy.deepcopy(model.encoder.state_dict())
    longitudinal_weight_before = model.actor.weight[1].detach().clone()
    longitudinal_bias_before = model.actor.bias[1].detach().clone()
    steering_before = model.actor.weight[0].detach().clone()
    critic_before = copy.deepcopy(model.critic.state_dict())
    log_std_before = model.log_std.detach().clone()

    result = distill_safety_actor(
        model, states, targets, intervened, severity, epochs=12, batch_size=32,
        learning_rate=8e-4, intervention_weight=2.0, steering_weight=1.0,
        train_encoder=False, steering_only=True, seed=27)

    assert result.final_intervention_mse < result.initial_intervention_mse
    assert result.encoder_changed_l1 == 0.0
    assert result.longitudinal_actor_changed_l1 == 0.0
    assert result.steering_actor_changed_l1 > 0
    for name, value in encoder_before.items():
        torch.testing.assert_close(model.encoder.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model.actor.weight[1], longitudinal_weight_before, rtol=0, atol=0)
    torch.testing.assert_close(model.actor.bias[1], longitudinal_bias_before, rtol=0, atol=0)
    assert not torch.equal(model.actor.weight[0], steering_before)
    for name, value in critic_before.items():
        torch.testing.assert_close(model.critic.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model.log_std, log_std_before, rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_component_only_modes_are_mutually_exclusive_and_require_frozen_encoder():
    model = DrivingPolicy('state', neighbors=1, hidden_size=16)
    states = np.zeros((4, 27), np.float32)
    targets = np.zeros((4, 2), np.float32)
    intervened = np.array([1, 0, 0, 0], bool)
    severity = np.array([.5, 0, 0, 0], np.float32)
    with pytest.raises(ValueError):
        distill_safety_actor(model, states, targets, intervened, severity,
                             train_encoder=False, longitudinal_only=True, steering_only=True)
    with pytest.raises(ValueError):
        distill_safety_actor(model, states, targets, intervened, severity,
                             train_encoder=True, steering_only=True)


def test_candidate_selector_can_be_explicitly_restricted_to_validation2():
    suite = [{'name':'s','metadrive_map':'S','traffic_density':.1,'target_speed':15.,
              'max_steps':200,'seeds':[11000,11001]}]
    rows = [
        dict(seed=11000,suite_stage='s',success=True,collision=False,out_of_road=False,truncated=False,route_completion=1.),
        dict(seed=11001,suite_stage='s',success=False,collision=True,out_of_road=False,truncated=False,route_completion=.4),
    ]
    report = {'suite': suite, 'episodes': rows}
    result = select_safety_candidate({'a':report,'b':copy.deepcopy(report)},
                                     allowed_seed_range=SPLITS['validation2'], selection_domain='validation2')
    assert result['selection_domain'] == 'validation2_seeds_11000_11199_only'
    with pytest.raises(ValueError):
        select_safety_candidate({'a':report,'b':copy.deepcopy(report)})


def test_expert_teacher_rejects_inactive_binding_before_importing_metadrive():
    from src.experts.metadrive_idm import MetaDriveIDMTeacher
    class Fake:
        agents = []
    with pytest.raises(ValueError):
        MetaDriveIDMTeacher(Fake(), 'agent0')


def test_path_suite_uses_curved_mixed_maps_and_longer_horizons():
    from src.evaluation.path_suite import standard_path_suite
    suite = standard_path_suite('validation2', 2)
    assert [s.name for s in suite] == ['curve_sparse','curve_moderate','mixed_sparse','mixed_moderate']
    assert all('C' in s.metadrive_map or 'X' in s.metadrive_map for s in suite)
    assert min(s.max_steps for s in suite) >= 420
    assert len({seed for s in suite for seed in s.seeds}) == 8


def test_anchored_expert_distillation_learns_teacher_without_touching_critic_or_reference():
    from src.experts.distillation import distill_anchored_expert
    torch.manual_seed(271)
    rng = np.random.default_rng(271)
    model = DrivingPolicy('state', neighbors=1, hidden_size=24)
    reference = copy.deepcopy(model)
    states = rng.normal(size=(192, 27)).astype(np.float32)
    with torch.no_grad():
        ref_actions = actor_actions(reference, torch.as_tensor(states)).cpu().numpy()
    targets = ref_actions.copy()
    targets[:, 0] = np.clip(targets[:, 0] + 0.35 * np.tanh(states[:, 3]), -1, 1)
    targets[:, 1] = np.clip(targets[:, 1] - 0.08 * np.maximum(states[:, 0], 0), -1, 1)
    intervened = np.abs(targets[:, 0] - ref_actions[:, 0]) > 0.02
    severity = np.clip(np.abs(states[:, 3]), 0, 1).astype(np.float32)
    critic_before = copy.deepcopy(model.critic.state_dict())
    log_std_before = model.log_std.detach().clone()
    reference_before = copy.deepcopy(reference.state_dict())

    result = distill_anchored_expert(
        model, reference, states, targets, intervened, severity,
        epochs=20, batch_size=48, learning_rate=5e-4,
        intervention_weight=2.0, longitudinal_teacher_weight=0.15,
        reference_action_coef=1.5, feature_anchor_coef=0.1, seed=271,
    )

    assert result.final_teacher_mse < result.initial_teacher_mse
    assert result.final_reference_mse < 0.03
    assert result.encoder_changed_l1 > 0
    assert result.steering_actor_changed_l1 > 0
    for name, value in critic_before.items():
        torch.testing.assert_close(model.critic.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model.log_std, log_std_before, rtol=0, atol=0)
    for name, value in reference_before.items():
        torch.testing.assert_close(reference.state_dict()[name], value, rtol=0, atol=0)


def test_anchored_expert_distillation_validates_alignment_and_weights():
    from src.experts.distillation import distill_anchored_expert
    model = DrivingPolicy('state', neighbors=1, hidden_size=16)
    reference = copy.deepcopy(model)
    states = np.zeros((8, 27), np.float32)
    targets = np.zeros((8, 2), np.float32)
    intervened = np.zeros(8, bool)
    severity = np.zeros(8, np.float32)
    with pytest.raises(ValueError):
        distill_anchored_expert(model, reference, states, targets[:-1], intervened, severity)
    with pytest.raises(ValueError):
        distill_anchored_expert(model, reference, states, targets, intervened, np.full(8, 1.1, np.float32))
    with pytest.raises(ValueError):
        distill_anchored_expert(model, reference, states, targets, intervened, severity,
                                reference_action_coef=-1)
