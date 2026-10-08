import numpy as np
import torch
from src.agents.distillation import distill_actor
from src.agents.policy import DrivingPolicy


def test_distillation_reduces_action_mse_and_changes_rgb_encoder():
    rng = np.random.default_rng(5)
    obs = {'rgb': rng.integers(0, 256, size=(24, 3, 40, 80), dtype=np.uint8),
           'motion': rng.normal(0, .3, size=(24, 20)).astype(np.float32)}
    targets = np.column_stack((np.tanh(obs['motion'][:, 0]), np.tanh(obs['motion'][:, 1]))).astype(np.float32)
    torch.manual_seed(5)
    model = DrivingPolicy('rgb_ego', hidden_size=32)
    result = distill_actor(model, obs, targets, epochs=10, batch_size=8, learning_rate=1e-3, seed=5)
    assert result.final_mse < result.initial_mse
    assert result.actor_encoder_changed_l1 > 0
