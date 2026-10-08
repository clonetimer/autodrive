"""Trainable local-state, semantic-BEV and raw-RGB encoders."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from src.envs.observation import state_dimension


class BEVEncoder(nn.Module):
    def __init__(self, features: int = 64, frames: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(5 * frames, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(32 * 3 * 3, features), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image.float() / 255.0)


class RGBEncoder(nn.Module):
    """Small CPU-friendly encoder for MetaDrive RGBCamera frames.

    The actor sees pixels plus ego/navigation motion only. No semantic raster or
    privileged neighbour kinematics are fused into this branch.
    """
    def __init__(self, features: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Linear(32 * 3 * 3, features), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image.float() / 255.0)


class ObservationEncoder(nn.Module):
    def __init__(self, mode: str, neighbors: int = 6, features: int = 64, frames: int = 1):
        super().__init__()
        self.mode = mode
        self.frames = frames
        self.output_dim = features
        if mode in {'state', 'hybrid'}:
            self.state = nn.Sequential(nn.Linear(state_dimension(neighbors), features), nn.Tanh(),
                                       nn.Linear(features, features), nn.Tanh())
        if mode in {'bev', 'hybrid', 'bev_ego'}:
            self.bev = BEVEncoder(features, frames)
            if frames > 1:
                self.temporal = nn.Sequential(nn.Linear(features + 2 * frames, features), nn.Tanh())
        if mode in {'bev_ego', 'rgb_ego'}:
            self.motion = nn.Sequential(nn.Linear(20, features), nn.Tanh(), nn.Linear(features, features), nn.Tanh())
        if mode == 'rgb_ego':
            self.rgb = RGBEncoder(features)
        if mode in {'hybrid', 'bev_ego', 'rgb_ego'}:
            self.fusion = nn.Sequential(nn.Linear(2 * features, features), nn.Tanh())

    def forward(self, observation):
        if self.mode == 'state':
            tensor = observation['state'] if isinstance(observation, dict) else observation
            return self.state(tensor.float())
        if self.mode == 'rgb_ego':
            pixels = self.rgb(observation['rgb'])
            motion = self.motion(observation['motion'].float())
            return self.fusion(torch.cat((motion, pixels), dim=-1))
        image_features = self.bev(observation['bev'])
        if self.frames > 1:
            image_features = self.temporal(torch.cat((image_features, observation['history'].float()), dim=-1))
        if self.mode == 'bev_ego':
            return self.fusion(torch.cat((self.motion(observation['motion'].float()), image_features), dim=-1))
        if self.mode == 'bev':
            return image_features
        return self.fusion(torch.cat((self.state(observation['state'].float()), image_features), dim=-1))


def stack_observations(items: list):
    if isinstance(items[0], dict):
        return {key: np.stack([item[key] for item in items]) for key in items[0]}
    return np.stack(items)


def zero_observation(example):
    if isinstance(example, dict):
        return {key: np.zeros_like(value) for key, value in example.items()}
    return np.zeros_like(example)


def tensor_observation(observation, device='cpu'):
    if isinstance(observation, dict):
        return {key: torch.as_tensor(value, device=device) for key, value in observation.items()}
    return torch.as_tensor(observation, device=device)


def index_observation(observation, indices):
    if isinstance(observation, dict):
        return {key: value[indices] for key, value in observation.items()}
    return observation[indices]
