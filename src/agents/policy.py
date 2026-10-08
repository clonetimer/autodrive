"""Squashed-Gaussian driving actor; optional centralized MAPPO value function."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from src.perception.encoder import ObservationEncoder, stack_observations, tensor_observation


class DrivingPolicy(nn.Module):
    def __init__(self, mode: str, neighbors: int = 6, hidden_size: int = 64, central_dim: int = 0, bev_frames: int = 1):
        super().__init__()
        self.encoder = ObservationEncoder(mode, neighbors, hidden_size, bev_frames)
        self.actor = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))
        self.central_dim = central_dim
        self.critic = (nn.Sequential(nn.Linear(central_dim, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
                       if central_dim else nn.Linear(hidden_size, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        value_layer = self.critic[-1] if isinstance(self.critic, nn.Sequential) else self.critic
        nn.init.orthogonal_(value_layer.weight, 1.0)

    def distribution(self, observation):
        features = self.encoder(observation)
        mean = self.actor(features)
        return Normal(mean, self.log_std.clamp(-5, 1).exp().expand_as(mean)), features

    @staticmethod
    def log_probability(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        # Numerically stable log(1 - tanh(z)^2), not an epsilon-clipped inverse.
        correction = 2 * (math.log(2) - latent - nn.functional.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(-1)

    def value(self, observation, central_state=None):
        if self.central_dim:
            if central_state is None or central_state.shape[-1] != self.central_dim:
                raise ValueError('MAPPO critic requires a correctly shaped centralized state')
            return self.critic(central_state.float()).squeeze(-1)
        return self.critic(self.encoder(observation)).squeeze(-1)

    def sample(self, observation, central_state=None, deterministic=False):
        distribution, features = self.distribution(observation)
        latent = distribution.mean if deterministic else distribution.rsample()
        action = latent.tanh()
        log_prob = self.log_probability(distribution, latent)
        value = self.critic(central_state.float()).squeeze(-1) if self.central_dim else self.critic(features).squeeze(-1)
        return action, latent, log_prob, value

    def evaluate(self, observation, latent, central_state=None):
        distribution, features = self.distribution(observation)
        log_prob = self.log_probability(distribution, latent)
        # Exploration regularizer is entropy of the underlying Gaussian, not the
        # analytically unavailable entropy of the tanh-transformed distribution.
        entropy = distribution.entropy().sum(-1)
        value = self.critic(central_state.float()).squeeze(-1) if self.central_dim else self.critic(features).squeeze(-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def predict(self, observation, deterministic=True) -> np.ndarray:
        device = next(self.parameters()).device
        batch = tensor_observation(stack_observations([observation]), device)
        distribution, _ = self.distribution(batch)
        latent = distribution.mean if deterministic else distribution.sample()
        return latent.tanh()[0].cpu().numpy().astype(np.float32)
