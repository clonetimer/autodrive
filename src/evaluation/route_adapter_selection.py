"""Scenario-level selection gate for learned lateral route adapters.

This gate intentionally evaluates *raw* learned policies with no RouteAssist or
SafetySupervisor.  It prevents a curve-specialist checkpoint from being
promoted by merely trading collisions for off-road failures or by depending on
runtime rules.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class RouteAdapterGate:
    min_success_gain_pp: float = 20.0
    min_catastrophic_reduction_pp: float = 10.0
    max_collision_increase_pp: float = 0.0
    max_offroad_increase_pp: float = 0.0
    min_curve_success: float = 2 / 3
    max_stage_success_drop_pp: float = 0.0


def _rates(episodes: Iterable[dict]) -> dict[str, float]:
    rows = list(episodes)
    if not rows:
        raise ValueError('selection needs at least one episode')
    n = float(len(rows))
    success = sum(bool(r.get('success')) for r in rows) / n
    collision = sum(bool(r.get('collision')) for r in rows) / n
    offroad = sum(bool(r.get('out_of_road')) for r in rows) / n
    catastrophic = sum(bool(r.get('collision')) or bool(r.get('out_of_road')) for r in rows) / n
    return {'episodes': int(n), 'success_rate': success, 'collision_rate': collision,
            'out_of_road_rate': offroad, 'catastrophic_rate': catastrophic}


def _stage_rates(episodes: Iterable[dict]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {}
    for row in episodes:
        groups.setdefault(str(row['suite_stage']), []).append(row)
    return {name: _rates(rows) for name, rows in sorted(groups.items())}


def select_route_adapter(baseline_episodes: Iterable[dict], candidate_episodes: Iterable[dict],
                         gate: RouteAdapterGate = RouteAdapterGate()) -> dict:
    baseline = list(baseline_episodes); candidate = list(candidate_episodes)
    base_keys = {(r['suite_stage'], int(r['seed'])) for r in baseline}
    cand_keys = {(r['suite_stage'], int(r['seed'])) for r in candidate}
    if base_keys != cand_keys:
        raise ValueError('baseline/candidate validation scenarios must match exactly')
    b, c = _rates(baseline), _rates(candidate)
    bs, cs = _stage_rates(baseline), _stage_rates(candidate)
    success_gain = 100 * (c['success_rate'] - b['success_rate'])
    catastrophic_reduction = 100 * (b['catastrophic_rate'] - c['catastrophic_rate'])
    collision_increase = 100 * (c['collision_rate'] - b['collision_rate'])
    offroad_increase = 100 * (c['out_of_road_rate'] - b['out_of_road_rate'])
    stage_drops = {name: 100 * (cs[name]['success_rate'] - bs[name]['success_rate']) for name in bs}
    checks = {
        'success_gain': success_gain >= gate.min_success_gain_pp,
        'catastrophic_reduction': catastrophic_reduction >= gate.min_catastrophic_reduction_pp,
        'collision_non_regression': collision_increase <= gate.max_collision_increase_pp,
        'offroad_non_regression': offroad_increase <= gate.max_offroad_increase_pp,
        'curve_success': cs.get('curve_tracking', {}).get('success_rate', 0.0) >= gate.min_curve_success,
        'stage_success_non_regression': all(delta >= -gate.max_stage_success_drop_pp for delta in stage_drops.values()),
    }
    return {'gate': asdict(gate), 'baseline': b, 'candidate': c, 'baseline_stages': bs,
            'candidate_stages': cs, 'deltas_pp': {'success': success_gain,
            'catastrophic_reduction': catastrophic_reduction, 'collision_increase': collision_increase,
            'offroad_increase': offroad_increase, 'stage_success': stage_drops},
            'checks': checks, 'promoted': all(checks.values())}
