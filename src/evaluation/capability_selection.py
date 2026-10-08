"""Validation-only selection for specialized MetaDrive capability policies."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
from src.evaluation.model_selection import _suite_signature


@dataclass(frozen=True)
class CapabilityScore:
    name: str
    scenarios: int
    success_rate: float
    catastrophic_rate: float
    collision_rate: float
    out_of_road_rate: float
    timeout_rate: float
    stall_time_fraction: float
    mean_route_completion: float

    def rank_key(self):
        return (-self.success_rate, self.catastrophic_rate, self.timeout_rate,
                self.stall_time_fraction, self.collision_rate,
                -self.mean_route_completion, self.name)

    def to_dict(self):
        return asdict(self)


def _score(name: str, report: dict) -> CapabilityScore:
    rows = report.get('episodes')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Capability report has no episode rows')
    n = len(rows)
    if len({(int(row['seed']), row.get('suite_stage')) for row in rows}) != n:
        raise ValueError('Capability selection expects one row per stage/seed scenario')
    return CapabilityScore(
        name=name, scenarios=n,
        success_rate=sum(bool(r.get('success')) for r in rows)/n,
        catastrophic_rate=sum(bool(r.get('collision')) or bool(r.get('out_of_road')) for r in rows)/n,
        collision_rate=sum(bool(r.get('collision')) for r in rows)/n,
        out_of_road_rate=sum(bool(r.get('out_of_road')) for r in rows)/n,
        timeout_rate=sum(bool(r.get('truncated')) and not bool(r.get('success')) for r in rows)/n,
        stall_time_fraction=sum(float(r.get('stall_time_fraction', 0.0))*float(r.get('duration_s', 1.0)) for r in rows)
                            / max(sum(float(r.get('duration_s', 1.0)) for r in rows), 1e-9),
        mean_route_completion=sum(float(r.get('route_completion', 0.0)) for r in rows)/n,
    )


def _stage_rates(report: dict) -> dict[str, dict]:
    result = {}
    for row in report['episodes']:
        result.setdefault(row['suite_stage'], []).append(row)
    return {stage: {
        'success_rate': sum(bool(r.get('success')) for r in rows)/len(rows),
        'catastrophic_rate': sum(bool(r.get('collision')) or bool(r.get('out_of_road')) for r in rows)/len(rows),
        'timeout_rate': sum(bool(r.get('truncated')) and not bool(r.get('success')) for r in rows)/len(rows),
        'mean_route_completion': sum(float(r.get('route_completion', 0.0)) for r in rows)/len(rows),
    } for stage, rows in result.items()}


def select_capability_candidate(reports: Mapping[str, dict], *, allowed_seed_range: tuple[int, int],
                                selection_domain: str='validation7') -> dict:
    if len(reports) < 2:
        raise ValueError('Provide at least two candidate reports')
    signatures = {_suite_signature(report, allowed_seed_range) for report in reports.values()}
    if len(signatures) != 1:
        raise ValueError('All candidates must use the exact same frozen capability suite')
    scores = sorted((_score(name, report) for name, report in reports.items()), key=lambda x:x.rank_key())
    return {
        'selection_domain': f'{selection_domain}_seeds_{allowed_seed_range[0]}_{allowed_seed_range[1]}_only',
        'ranking_rule': ['maximize_success_rate','minimize_collision_or_offroad_rate','minimize_timeout_rate',
                         'minimize_stall_time_fraction','minimize_collision_rate','maximize_mean_route_completion'],
        'selected': scores[0].name,
        'scores': [x.to_dict() for x in scores],
        'stage_metrics': {name:_stage_rates(report) for name,report in reports.items()},
        'suite_signature': [list(x) for x in next(iter(signatures))],
        'claim_boundary': 'Capability selection is validation-only and is not a safety certificate.',
    }


def capability_promotion_decision(selection: dict, baseline: str, *, min_success_gain: float=.05,
                                  min_catastrophic_reduction: float=.10,
                                  min_timeout_reduction: float=.10,
                                  max_collision_increase: float=.05,
                                  max_offroad_increase: float=.05,
                                  max_stage_catastrophic_increase: float=.20) -> dict:
    scores = {r['name']:r for r in selection['scores']}
    candidate_name = selection['selected']
    if baseline not in scores or candidate_name not in scores:
        raise ValueError('Baseline/selected candidate is missing')
    c,b=scores[candidate_name],scores[baseline]
    deltas={
        'success_gain':c['success_rate']-b['success_rate'],
        'catastrophic_reduction':b['catastrophic_rate']-c['catastrophic_rate'],
        'timeout_reduction':b['timeout_rate']-c['timeout_rate'],
        'collision_increase':c['collision_rate']-b['collision_rate'],
        'offroad_increase':c['out_of_road_rate']-b['out_of_road_rate'],
        'stall_reduction':b['stall_time_fraction']-c['stall_time_fraction'],
        'route_completion_gain':c['mean_route_completion']-b['mean_route_completion'],
    }
    stage_c=selection['stage_metrics'][candidate_name]; stage_b=selection['stage_metrics'][baseline]
    stage_regressions={stage:stage_c[stage]['catastrophic_rate']-stage_b[stage]['catastrophic_rate'] for stage in stage_b}
    checks={
        'material_gain': (deltas['success_gain'] >= min_success_gain or
                          deltas['catastrophic_reduction'] >= min_catastrophic_reduction or
                          deltas['timeout_reduction'] >= min_timeout_reduction),
        'collision_bound': deltas['collision_increase'] <= max_collision_increase,
        'offroad_bound': deltas['offroad_increase'] <= max_offroad_increase,
        'stage_catastrophic_bound': max(stage_regressions.values(), default=0.0) <= max_stage_catastrophic_increase,
    }
    return {
        'baseline':baseline,'candidate':candidate_name,'promoted':all(checks.values()),
        'deltas':deltas,'stage_catastrophic_increase':stage_regressions,'checks':checks,
        'thresholds':{'min_success_gain':min_success_gain,'min_catastrophic_reduction':min_catastrophic_reduction,
                      'min_timeout_reduction':min_timeout_reduction,'max_collision_increase':max_collision_increase,
                      'max_offroad_increase':max_offroad_increase,
                      'max_stage_catastrophic_increase':max_stage_catastrophic_increase},
        'decision_domain':f"{selection.get('selection_domain','validation_only')}_before_fresh_release",
        'claim_boundary':'Promotion is an engineering registry gate, not a statistical safety guarantee.',
    }
