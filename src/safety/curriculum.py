"""Curriculum definitions for collecting safety-supervisor corrections.

The curriculum only changes *training/collection* environments.  Frozen release
splits are evaluated separately so scenario difficulty cannot leak test outcomes
back into policy selection.  Stages intentionally use MetaDrive block strings
that are supported by the pinned 0.4.3 simulator (straight, curve and mixed
procedural roads) and progressively increase traffic interaction.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import yaml
from src.utils.config import EnvConfig


@dataclass(frozen=True)
class SafetyCollectionStage:
    name: str
    metadrive_map: str
    traffic_density: float
    target_speed: float = 15.0
    max_steps: int = 300

    def __post_init__(self):
        if not self.name or not self.metadrive_map:
            raise ValueError('Safety curriculum stage requires a name and MetaDrive map')
        if not 0 <= self.traffic_density <= 1:
            raise ValueError('Safety curriculum traffic density must be in [0,1]')
        if self.target_speed <= 0 or self.max_steps <= 0:
            raise ValueError('Safety curriculum speed/horizon must be positive')

    def apply(self, base: EnvConfig) -> EnvConfig:
        if base.backend != 'metadrive' or base.num_agents != 1 or base.observation != 'state':
            raise ValueError('Safety collection curriculum requires single-agent MetaDrive state input')
        return replace(base, metadrive_map=self.metadrive_map,
                       metadrive_traffic_density=float(self.traffic_density),
                       target_speed=float(self.target_speed), max_steps=int(self.max_steps))

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_SAFETY_CURRICULUM = (
    SafetyCollectionStage('straight_moderate', 'S', 0.18, 15.0, 300),
    SafetyCollectionStage('straight_dense', 'SSS', 0.32, 14.0, 320),
    SafetyCollectionStage('curve_dense', 'SCS', 0.26, 13.0, 340),
)


def load_safety_curriculum(path: str | Path | None = None) -> tuple[SafetyCollectionStage, ...]:
    if path is None:
        return DEFAULT_SAFETY_CURRICULUM
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    stages = payload.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('Safety curriculum YAML must contain a non-empty stages list')
    allowed = {'name', 'metadrive_map', 'traffic_density', 'target_speed', 'max_steps'}
    result = []
    for raw in stages:
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError('Invalid safety curriculum stage fields')
        result.append(SafetyCollectionStage(**raw))
    if len({stage.name for stage in result}) != len(result):
        raise ValueError('Safety curriculum stage names must be unique')
    return tuple(result)
