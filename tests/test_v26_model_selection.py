import copy
import pytest
from src.evaluation.model_selection import promotion_decision, score_candidate, select_safety_candidate


def _report(rows, seeds=(1000, 1001)):
    return {
        'suite': [{'name': 's', 'metadrive_map': 'S', 'traffic_density': .1,
                   'target_speed': 15., 'max_steps': 200, 'seeds': list(seeds)}],
        'episodes': rows,
    }


def _rows(successes, collisions, offroads=None, completions=None):
    offroads = offroads or [False] * len(successes)
    completions = completions or [1. if x else .5 for x in successes]
    return [dict(seed=1000+i, suite_stage='s', success=s, collision=c,
                 out_of_road=o, truncated=not s and not c and not o,
                 route_completion=r)
            for i,(s,c,o,r) in enumerate(zip(successes,collisions,offroads,completions))]


def test_selector_prefers_success_then_catastrophic_rate():
    base = _report(_rows([1,0],[0,1]))
    safer = _report(_rows([1,0],[0,0],[0,1]))  # same catastrophic rate as base
    # Better completion breaks the remaining tie only after success/catastrophic/collision/timeout.
    safer['episodes'][1]['truncated'] = False
    safer['episodes'][1]['route_completion'] = .7
    result = select_safety_candidate({'base':base,'safer':safer})
    assert result['selected'] == 'safer'
    assert result['scores'][0]['success_rate'] == .5


def test_selector_rejects_release_seed_leakage_and_mismatched_suite():
    report = _report(_rows([1,0],[0,1]))
    leak = copy.deepcopy(report)
    leak['suite'][0]['seeds'] = [9000,9001]
    with pytest.raises(ValueError):
        select_safety_candidate({'a':report,'b':leak})
    mismatch = copy.deepcopy(report)
    mismatch['suite'][0]['traffic_density'] = .2
    with pytest.raises(ValueError):
        select_safety_candidate({'a':report,'b':mismatch})


def test_score_uses_collision_or_offroad_not_sum():
    rows = _rows([0,1],[1,0],[1,0])
    score = score_candidate('x', _report(rows))
    assert score.catastrophic_rate == .5
    assert score.collision_rate == .5 and score.out_of_road_rate == .5


def test_promotion_gate_rejects_offroad_tradeoff_without_material_gain():
    base = _report(_rows([1,0],[0,1],[0,0]))
    candidate = _report(_rows([1,0],[0,0],[0,1], completions=[1.0,.8]))
    selection = select_safety_candidate({'base':base,'candidate':candidate})
    decision = promotion_decision(selection, 'base', min_success_gain=.05,
                                  min_catastrophic_reduction=.10, max_offroad_increase=.05)
    assert selection['selected'] == 'candidate'
    assert not decision['promoted']
    assert 'no_material_success_or_catastrophic_improvement' in decision['reasons'] or 'offroad_regression' in decision['reasons']


def test_promotion_gate_accepts_material_catastrophic_reduction_without_regression():
    base_rows = _rows([1,0,0,0],[0,1,1,0],[0,0,0,1], completions=[1,.4,.4,.4])
    candidate_rows = _rows([1,1,0,0],[0,0,1,0],[0,0,0,1], completions=[1,1,.5,.5])
    # Fix generated seeds and suite to four scenarios.
    suite = {'suite':[{'name':'s','metadrive_map':'S','traffic_density':.1,'target_speed':15.,
                       'max_steps':200,'seeds':[1000,1001,1002,1003]}]}
    base = dict(suite, episodes=base_rows); candidate = dict(suite, episodes=candidate_rows)
    selection = select_safety_candidate({'base':base,'candidate':candidate})
    decision = promotion_decision(selection, 'base', min_success_gain=.20,
                                  min_catastrophic_reduction=.20, max_offroad_increase=.05)
    assert decision['promoted']
    assert decision['checks']['material_gain']
