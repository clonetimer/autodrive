"""Leakage-resistant safety candidate selection on one frozen validation suite.

Selection is deliberately deterministic and uses *raw-policy* outcomes only.
Release/final seed domains must never be passed here.  The ranking is
lexicographic: maximize success, then minimize catastrophic outcomes
(collision OR off-road), then collisions, then timeouts, then maximize route
completion.  Promotion is a second, predeclared gate relative to a baseline;
winning the ranking is not enough to enter the curated model registry.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping


@dataclass(frozen=True)
class CandidateScore:
    name: str
    scenarios: int
    success_rate: float
    catastrophic_rate: float
    collision_rate: float
    out_of_road_rate: float
    timeout_rate: float
    mean_route_completion: float

    def rank_key(self):
        return (-self.success_rate, self.catastrophic_rate, self.collision_rate,
                self.timeout_rate, -self.mean_route_completion, self.name)

    def to_dict(self):
        return asdict(self)


def _suite_signature(report: dict, allowed_seed_range: tuple[int, int] = (1000, 1199)) -> tuple:
    suite = report.get('suite')
    if not isinstance(suite, list) or not suite:
        raise ValueError('Safety selection requires suite_report.json inputs')
    signature = []
    for spec in suite:
        seeds = tuple(int(x) for x in spec.get('seeds', ()))
        if not seeds:
            raise ValueError('Safety suite stage is missing seeds')
        lo, hi = allowed_seed_range
        if any(seed < lo or seed > hi for seed in seeds):
            raise ValueError(f'Candidate selection may only use seeds {lo}-{hi}')
        signature.append((spec.get('name'), spec.get('metadrive_map'),
                          float(spec.get('traffic_density')), float(spec.get('target_speed')),
                          int(spec.get('max_steps')), seeds))
    return tuple(signature)


def score_candidate(name: str, report: dict) -> CandidateScore:
    episodes = report.get('episodes')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('Safety report has no episode rows')
    # Single-agent safety suite: one row per independent scenario.
    if len({(int(row['seed']), row.get('suite_stage')) for row in episodes}) != len(episodes):
        raise ValueError('Safety selection expects one agent episode per unique stage/seed scenario')
    n = len(episodes)
    success = sum(bool(row.get('success')) for row in episodes) / n
    collision = sum(bool(row.get('collision')) for row in episodes) / n
    offroad = sum(bool(row.get('out_of_road')) for row in episodes) / n
    timeout = sum(bool(row.get('truncated')) and not bool(row.get('success')) for row in episodes) / n
    catastrophic = sum(bool(row.get('collision')) or bool(row.get('out_of_road')) for row in episodes) / n
    completion = sum(float(row.get('route_completion', 0.0)) for row in episodes) / n
    return CandidateScore(name, n, success, catastrophic, collision, offroad, timeout, completion)


def promotion_decision(selection: dict, baseline: str, *,
                       min_success_gain: float = 0.05,
                       min_catastrophic_reduction: float = 0.10,
                       max_offroad_increase: float = 0.05,
                       max_collision_increase: float = 0.0) -> dict:
    """Apply a predeclared validation-only promotion gate.

    A candidate must demonstrate a material improvement in either success or
    catastrophic outcomes, without buying that improvement through a material
    off-road regression or a collision increase.  This gate is intentionally
    conservative because release/final seeds are not allowed to influence
    model promotion.
    """
    scores = {row['name']: row for row in selection.get('scores', [])}
    selected_name = selection.get('selected')
    if selected_name not in scores:
        raise ValueError('Selection is missing the selected candidate score')
    if baseline not in scores:
        raise ValueError(f'Baseline {baseline!r} is not present in candidate scores')
    candidate = scores[selected_name]
    base = scores[baseline]
    deltas = {
        'success_gain': candidate['success_rate'] - base['success_rate'],
        'catastrophic_reduction': base['catastrophic_rate'] - candidate['catastrophic_rate'],
        'collision_increase': candidate['collision_rate'] - base['collision_rate'],
        'offroad_increase': candidate['out_of_road_rate'] - base['out_of_road_rate'],
        'timeout_increase': candidate['timeout_rate'] - base['timeout_rate'],
        'route_completion_gain': candidate['mean_route_completion'] - base['mean_route_completion'],
    }
    material_gain = (deltas['success_gain'] >= min_success_gain or
                     deltas['catastrophic_reduction'] >= min_catastrophic_reduction)
    constraints = {
        'material_gain': material_gain,
        'collision_non_regression': deltas['collision_increase'] <= max_collision_increase,
        'offroad_non_regression': deltas['offroad_increase'] <= max_offroad_increase,
    }
    promoted = all(constraints.values())
    reasons = []
    if not constraints['material_gain']:
        reasons.append('no_material_success_or_catastrophic_improvement')
    if not constraints['collision_non_regression']:
        reasons.append('collision_regression')
    if not constraints['offroad_non_regression']:
        reasons.append('offroad_regression')
    return {
        'baseline': baseline,
        'candidate': selected_name,
        'promoted': promoted,
        'thresholds': {
            'min_success_gain': min_success_gain,
            'min_catastrophic_reduction': min_catastrophic_reduction,
            'max_collision_increase': max_collision_increase,
            'max_offroad_increase': max_offroad_increase,
        },
        'deltas': deltas,
        'checks': constraints,
        'reasons': reasons,
        'decision_domain': 'validation_only_before_release_evaluation',
        'claim_boundary': ('Promotion is an engineering model-registry gate, not a statistical '
                           'safety guarantee or certification.'),
    }


def select_safety_candidate(reports: Mapping[str, dict], *, allowed_seed_range: tuple[int, int] = (1000, 1199), selection_domain: str = 'validation') -> dict:
    if len(reports) < 2:
        raise ValueError('Provide at least two candidate reports')
    if len(allowed_seed_range) != 2 or allowed_seed_range[0] > allowed_seed_range[1]:
        raise ValueError('Invalid candidate-selection seed range')
    signatures = {_suite_signature(report, allowed_seed_range) for report in reports.values()}
    if len(signatures) != 1:
        raise ValueError('All candidates must use the exact same frozen validation suite')
    scores = [score_candidate(name, report) for name, report in reports.items()]
    ordered = sorted(scores, key=lambda item: item.rank_key())
    return {
        'selection_domain': f'{selection_domain}_seeds_{allowed_seed_range[0]}_{allowed_seed_range[1]}_only',
        'ranking_rule': ['maximize_success_rate', 'minimize_collision_or_offroad_rate',
                         'minimize_collision_rate', 'minimize_timeout_rate',
                         'maximize_mean_route_completion', 'name_tiebreak'],
        'selected': ordered[0].name,
        'scores': [item.to_dict() for item in ordered],
        'suite_signature': [list(item) for item in next(iter(signatures))],
        'claim_boundary': 'Selection result is not a safety certificate and may not be re-ranked using release outcomes.',
    }
