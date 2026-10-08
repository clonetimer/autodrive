"""Multi-condition MetaDrive safety evaluation with frozen seed domains.

A suite varies road geometry and traffic density while assigning a unique seed
to every scenario.  Aggregation therefore clusters by true scenario rather than
mistakenly treating the same seed under different maps as one sample.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from src.evaluation.evaluator import evaluate
from src.evaluation.metrics import summarize
from src.scenarios.generator import split_seeds
from src.utils.config import EnvConfig
from src.utils.logger import save_json


@dataclass(frozen=True)
class SafetySuiteSpec:
    name: str
    metadrive_map: str
    traffic_density: float
    target_speed: float
    max_steps: int
    seeds: tuple[int, ...]

    def __post_init__(self):
        if not self.name or not self.metadrive_map or not self.seeds:
            raise ValueError('Safety suite spec requires name, map and seeds')
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Safety suite seeds must be unique within a stage')
        if not 0 <= self.traffic_density <= 1 or self.target_speed <= 0 or self.max_steps <= 0:
            raise ValueError('Invalid safety suite stage parameters')

    def env(self, base: EnvConfig) -> EnvConfig:
        if base.backend != 'metadrive' or base.num_agents != 1 or base.observation != 'state':
            raise ValueError('Safety suite currently requires single-agent MetaDrive state input')
        return replace(base, metadrive_map=self.metadrive_map,
                       metadrive_traffic_density=float(self.traffic_density),
                       target_speed=float(self.target_speed), max_steps=int(self.max_steps))

    def to_dict(self) -> dict:
        result = asdict(self)
        result['seeds'] = list(self.seeds)
        return result


def standard_safety_suite(split: str, seeds_per_stage: int = 10) -> tuple[SafetySuiteSpec, ...]:
    if seeds_per_stage <= 0:
        raise ValueError('seeds_per_stage must be positive')
    all_seeds = split_seeds(split, 4 * seeds_per_stage)
    chunks = [tuple(all_seeds[i * seeds_per_stage:(i + 1) * seeds_per_stage]) for i in range(4)]
    return (
        SafetySuiteSpec('straight_moderate', 'S', .18, 15.0, 300, chunks[0]),
        SafetySuiteSpec('straight_dense', 'SSS', .35, 14.0, 320, chunks[1]),
        SafetySuiteSpec('curve_dense', 'SCS', .30, 13.0, 340, chunks[2]),
        SafetySuiteSpec('mixed_unseen', 'SCXTO', .22, 12.0, 360, chunks[3]),
    )


def evaluate_safety_suite(agent_factory, base_env: EnvConfig, suite: tuple[SafetySuiteSpec, ...],
                          output: str | Path | None = None, label: str = 'agent') -> dict:
    if not suite:
        raise ValueError('Safety suite must contain at least one stage')
    all_seeds = [seed for spec in suite for seed in spec.seeds]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError('Safety suite must assign a unique seed to every scenario')
    root = Path(output) if output else None
    stage_reports, episodes = [], []
    total_steps, total_elapsed = 0, 0.0
    for spec in suite:
        stage_output = root / spec.name if root else None
        report = evaluate(agent_factory, spec.env(base_env), list(spec.seeds), output=stage_output,
                          label=f'{label}:{spec.name}')
        for row in report['episodes']:
            episodes.append({**row, 'suite_stage': spec.name, 'metadrive_map': spec.metadrive_map,
                             'traffic_density': spec.traffic_density})
        stage_reports.append({'spec': spec.to_dict(), 'metrics': report['metrics'],
                              'world_steps': report['world_steps'], 'elapsed_seconds': report['elapsed_seconds']})
        total_steps += int(report['world_steps'])
        total_elapsed += float(report['elapsed_seconds'])
    aggregate = summarize(episodes)
    result = {'label': label, 'suite': [spec.to_dict() for spec in suite],
              'metrics': aggregate, 'stages': stage_reports, 'episodes': episodes,
              'world_steps': total_steps, 'elapsed_seconds': total_elapsed,
              'closed_loop_world_steps_per_second': total_steps / max(total_elapsed, 1e-9)}
    if root:
        root.mkdir(parents=True, exist_ok=True)
        save_json(result, root / 'suite_report.json')
    return result
