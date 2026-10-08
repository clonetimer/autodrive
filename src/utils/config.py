"""Validated YAML configuration; the saved resolved configuration is authoritative."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import copy
import yaml


@dataclass
class EnvConfig:
    backend: str = 'native'
    num_agents: int = 1
    observation: str = 'state'
    reward_source: str = 'official'  # MetaDrive: official simulator reward or project custom reward.
    rgb_width: int = 96
    rgb_height: int = 48
    max_steps: int = 400
    dt: float = 0.1
    substeps: int = 2
    road_length: float = 350.0
    lanes: int = 3
    lane_width: float = 3.6
    traffic_vehicles: int = 8
    target_speed: float = 15.0
    initial_speed: float = 8.0
    curve_amplitude: float = 0.0
    curve_wavelength: float = 220.0
    randomize: bool = False
    observation_noise: float = 0.0
    neighbors: int = 6
    state_schema_version: int = 3  # v3=legacy x-matched preview; v4=arc-length route preview.
    bev_size: int = 48
    bev_frames: int = 1
    reward_version: int = 3  # v2 remains explicit for historical-model evaluation.
    reward: str = 'balanced'
    seed: int = 0
    scenario: str = 'cruise'
    metadrive_map: str = 'S'
    metadrive_traffic_density: float = 0.1
    dataset: str | None = None
    safety_supervisor: bool = False
    safety_ttc_threshold_s: float = 1.5
    safety_risk_threshold: float = 0.75
    route_assist: bool = False
    route_assist_blend: float = 1.0
    route_assist_mode: str = 'fixed'
    route_assist_min_blend: float = 0.0
    route_assist_curvature_start: float = 0.06
    route_assist_curvature_full: float = 0.28
    route_assist_heading_start_rad: float = 0.035
    route_assist_heading_full_rad: float = 0.18
    route_assist_lane_start: float = 0.06
    route_assist_lane_full: float = 0.30

    def __post_init__(self):
        if self.backend not in {'native', 'metadrive', 'replay'}:
            raise ValueError(f'Unsupported backend {self.backend!r}; no implicit fallback is allowed.')
        if self.observation not in {'state', 'bev', 'hybrid', 'bev_ego', 'rgb_ego'}:
            raise ValueError('observation must be state, bev, hybrid, bev_ego or rgb_ego')
        if self.reward_source not in {'official', 'custom'}:
            raise ValueError('reward_source must be official or custom')
        if self.backend != 'metadrive' and self.reward_source != 'official':
            # Native already uses the project reward; the selector only disambiguates MetaDrive.
            raise ValueError('reward_source selector is MetaDrive-only')
        if self.observation == 'rgb_ego' and self.backend != 'metadrive':
            raise ValueError('rgb_ego requires the real MetaDrive backend')
        if not (32 <= self.rgb_width <= 512 and 32 <= self.rgb_height <= 512):
            raise ValueError('RGB dimensions must be between 32 and 512 pixels')
        if not 1 <= self.num_agents <= 16:
            raise ValueError('num_agents must be between 1 and 16')
        if not (0 < self.dt <= 0.2 and 1 <= self.substeps <= 20):
            raise ValueError('Invalid physical time step or substeps')
        if not (self.max_steps > 0 and self.road_length >= 50 and self.lanes >= 1 and self.lane_width >= 2.5):
            raise ValueError('Invalid road dimensions or horizon')
        if self.traffic_vehicles < 0 or self.target_speed <= 0 or self.initial_speed < 0:
            raise ValueError('Invalid traffic configuration')
        if self.bev_size < 32 or self.neighbors < 1 or self.observation_noise < 0:
            raise ValueError('Invalid observation configuration')
        if self.state_schema_version not in {3, 4}:
            raise ValueError('state_schema_version must be 3 (legacy) or 4 (arc-length route preview)')
        if self.reward_version not in {2, 3, 4}:
            raise ValueError('reward_version must be 2 (legacy), 3 (bounded risk-aware) or 4 (emergency-aware)')
        if not isinstance(self.bev_frames, int) or not 1 <= self.bev_frames <= 8:
            raise ValueError('bev_frames must be an integer from 1 to 8')
        if self.observation in {'state', 'rgb_ego'} and self.bev_frames != 1:
            raise ValueError('state/rgb_ego observations do not use semantic-BEV history')
        if self.reward not in {'balanced', 'safe', 'efficient'}:
            raise ValueError('Unknown reward profile')
        if self.scenario not in {'cruise', 'follow', 'overtake', 'emergency'}:
            raise ValueError('Unknown scenario')
        if self.safety_supervisor and self.backend != 'metadrive':
            raise ValueError('The runtime safety supervisor is currently MetaDrive-only')
        if self.route_assist and (self.backend != 'metadrive' or self.observation != 'state'):
            raise ValueError('route_assist currently requires MetaDrive state observations')
        if self.route_assist_mode not in {'fixed', 'adaptive'}:
            raise ValueError('route_assist_mode must be fixed or adaptive')
        if not 0 <= self.route_assist_min_blend <= self.route_assist_blend <= 1:
            raise ValueError('route assist blends must satisfy 0 <= min <= max <= 1')
        for start, full in ((self.route_assist_curvature_start, self.route_assist_curvature_full),
                            (self.route_assist_heading_start_rad, self.route_assist_heading_full_rad),
                            (self.route_assist_lane_start, self.route_assist_lane_full)):
            if not 0 <= start < full:
                raise ValueError('adaptive route-assist thresholds require 0 <= start < full')
        if not (self.safety_ttc_threshold_s > 0 and 0 < self.safety_risk_threshold <= 1):
            raise ValueError('Invalid safety-supervisor thresholds')


@dataclass
class TrainConfig:
    algorithm: str = 'ppo'
    total_steps: int = 16384  # Joint world steps, NOT number of active-agent transitions.
    rollout_steps: int = 256
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.04
    hidden_size: int = 64
    checkpoint_every: int = 4096
    seed: int = 0
    threads: int = 1
    device: str = 'cpu'
    reference_checkpoint: str | None = None
    reference_kl_coef: float = 0.0
    curriculum: bool = False
    curriculum_min_episodes: int = 20
    curriculum_success_threshold: float = 0.75
    curriculum_max_collision_rate: float = 0.15
    learning_starts: int = 256
    replay_max_bytes: int = 536870912  # Fail before allocating an oversized pixel buffer.
    replay_size: int = 50000
    tau: float = 0.005
    env: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        if isinstance(self.env, dict):
            self.env = EnvConfig(**self.env)
        if self.algorithm not in {'ppo', 'mappo', 'ippo', 'sac'}:
            raise ValueError('algorithm must be ppo, ippo, mappo or sac')
        if min(self.total_steps, self.rollout_steps, self.epochs, self.batch_size, self.threads) <= 0:
            raise ValueError('Training counts must be positive')
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.learning_rate > 0):
            raise ValueError('Invalid optimizer/return parameters')
        if self.curriculum and (self.env.backend != 'native' or self.algorithm == 'sac'):
            raise ValueError('Measured curriculum currently supports native PPO/IPPO/MAPPO only')
        if not (1 <= self.curriculum_min_episodes <= 40 and 0 <= self.curriculum_success_threshold <= 1
                and 0 <= self.curriculum_max_collision_rate <= 1):
            raise ValueError('Invalid curriculum thresholds or history size')
        if self.algorithm == 'sac' and self.env.num_agents != 1:
            raise ValueError('SAC currently supports one learning agent only')
        if self.reference_kl_coef < 0:
            raise ValueError('reference_kl_coef must be nonnegative')
        if self.reference_kl_coef > 0 and not self.reference_checkpoint:
            raise ValueError('reference_checkpoint is required when reference_kl_coef > 0')
        if self.reference_kl_coef > 0 and self.algorithm == 'sac':
            raise ValueError('reference-policy KL is implemented for PPO/IPPO/MAPPO only')
        if not (0 < self.tau <= 1 and self.replay_size > self.batch_size and self.checkpoint_every > 0):
            raise ValueError('Invalid buffer, target update or checkpoint setting')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deep_merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _load_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError(f'Cyclic configuration include: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a mapping')
    includes = data.pop('include', [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ValueError('include must be a list of relative YAML file paths')
    combined = {}
    for item in includes:
        child = Path(item)
        if child.is_absolute():
            raise ValueError('Configuration includes must be relative to the containing file')
        combined = deep_merge(combined, _load_yaml(path.parent / child, stack + (path,)))
    return deep_merge(combined, data)


def load_config(path: str | Path, overrides: dict | None = None) -> TrainConfig:
    return TrainConfig(**deep_merge(_load_yaml(Path(path)), overrides or {}))


def save_config(config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding='utf-8')
