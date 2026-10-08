"""Fresh MetaDrive capability suite for v2.8+ specialization experiments.

The suite separates five behaviors that were confounded in earlier aggregate
safety tests: curve tracking, ramp merging, bottleneck yielding, intersection
handling, and dense-traffic progress.  Each scenario receives a unique seed
from a fresh split.  The benchmark is still a finite research evaluation, not
a safety certification test.
"""
from __future__ import annotations
from src.evaluation.safety_suite import SafetySuiteSpec
from src.scenarios.generator import split_seeds


def standard_capability_suite(split: str, seeds_per_stage: int = 5) -> tuple[SafetySuiteSpec, ...]:
    if seeds_per_stage <= 0:
        raise ValueError('seeds_per_stage must be positive')
    all_seeds = split_seeds(split, 5 * seeds_per_stage)
    chunks = [tuple(all_seeds[i * seeds_per_stage:(i + 1) * seeds_per_stage]) for i in range(5)]
    return (
        SafetySuiteSpec('curve_tracking', 'SCS', .10, 10.0, 650, chunks[0]),
        SafetySuiteSpec('ramp_merge', 'SrS', .08, 9.0, 750, chunks[1]),
        SafetySuiteSpec('bottleneck_yield', 'SyS', .10, 9.0, 700, chunks[2]),
        SafetySuiteSpec('intersection_yield', 'SXT', .04, 8.0, 700, chunks[3]),
        SafetySuiteSpec('dense_progress', 'SSS', .20, 10.0, 700, chunks[4]),
    )
