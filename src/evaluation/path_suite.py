"""Path-following benchmark that separates lateral competence from dense-traffic deadlock.

The safety suite intentionally stresses traffic interaction and short horizons.
This companion suite uses curved/mixed maps, lighter traffic and longer horizons
so off-road/path failures can be diagnosed without interpreting congestion
timeouts as lateral-control failures.  It is still a closed-loop MetaDrive
benchmark, not a safety certification test.
"""
from __future__ import annotations
from src.evaluation.safety_suite import SafetySuiteSpec
from src.scenarios.generator import split_seeds


def standard_path_suite(split: str, seeds_per_stage: int = 10) -> tuple[SafetySuiteSpec, ...]:
    if seeds_per_stage <= 0:
        raise ValueError('seeds_per_stage must be positive')
    all_seeds = split_seeds(split, 4 * seeds_per_stage)
    chunks = [tuple(all_seeds[i * seeds_per_stage:(i + 1) * seeds_per_stage]) for i in range(4)]
    return (
        SafetySuiteSpec('curve_sparse', 'SCS', .05, 10.0, 420, chunks[0]),
        SafetySuiteSpec('curve_moderate', 'SCS', .15, 10.0, 440, chunks[1]),
        SafetySuiteSpec('mixed_sparse', 'SCXTO', .05, 10.0, 460, chunks[2]),
        SafetySuiteSpec('mixed_moderate', 'SCXTO', .15, 10.0, 480, chunks[3]),
    )
