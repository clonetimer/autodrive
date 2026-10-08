"""Interaction-focused MetaDrive suite for factorized longitudinal policies.

The standard capability suite remains the regression gate for route following.
This suite deliberately raises traffic density on merge/yield/progress scenarios
so candidate selection has sensitivity to longitudinal interaction behavior.
It uses a disjoint seed block within the same named split (offset 50 by default),
therefore standard capability and interaction results are not repeated scenarios.
"""
from __future__ import annotations

from src.evaluation.safety_suite import SafetySuiteSpec
from src.scenarios.generator import split_seed_block


def interaction_stress_suite(split: str, seeds_per_stage: int = 3,
                             seed_offset: int = 50) -> tuple[SafetySuiteSpec, ...]:
    if seeds_per_stage <= 0:
        raise ValueError('seeds_per_stage must be positive')
    if seed_offset < 0:
        raise ValueError('seed_offset must be nonnegative')
    all_seeds = split_seed_block(split, seed_offset, 4 * seeds_per_stage)
    chunks = [tuple(all_seeds[i * seeds_per_stage:(i + 1) * seeds_per_stage]) for i in range(4)]
    return (
        SafetySuiteSpec('ramp_merge_stress', 'SrS', .14, 9.0, 800, chunks[0]),
        SafetySuiteSpec('bottleneck_stress', 'SyS', .17, 9.0, 800, chunks[1]),
        SafetySuiteSpec('intersection_stress', 'SXT', .08, 8.5, 800, chunks[2]),
        SafetySuiteSpec('dense_progress_stress', 'SSS', .28, 10.0, 800, chunks[3]),
    )
