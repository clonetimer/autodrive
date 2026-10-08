"""Release-evidence invariants for the v2.6 safety-curriculum research release."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def test_v26_selection_is_validation_only_and_not_promoted():
    result = _load('reports/v2_6/candidate_selection.json')
    assert result['selection_domain'] == 'validation_seeds_1000_1199_only'
    seeds = [int(seed) for stage in result['suite_signature'] for seed in stage[-1]]
    assert seeds == list(range(1000, 1020))
    assert result['selected'] == 'dagger_balanced'
    assert result['promotion']['promoted'] is False
    assert 'offroad_regression' in result['promotion']['reasons']
    assert all(seed < 9000 for seed in seeds)


def test_v26_release5_is_frozen_consumed_and_diagnostic_only():
    report = _load('reports/v2_6/comparison.json')
    assert report['release5_consumed'] is True
    assert report['release5']['seeds'] == list(range(9000, 9040))
    assert report['selection']['release5_not_used_for_selection_or_promotion'] is True
    assert report['selection']['promotion']['promoted'] is False
    assert not set(report['selection']['validation_seeds']) & set(report['release5']['seeds'])
    # The candidate lowers collision but trades part of it for off-road failure;
    # release5 is evidence, not grounds to overturn the validation gate.
    raw = report['raw_policy']
    assert raw['collision_change_pp'] < 0
    assert raw['offroad_change_pp'] > 0
    assert raw['paired_success']['exact_mcnemar_p_two_sided'] == 1.0


def test_v26_longitudinal_supervisor_reduces_collision_but_not_lateral_failure():
    report = _load('reports/v2_6/comparison.json')
    sup = report['supervised_system']
    assert sup['selected']['collision_rate'] == 0.0
    assert sup['selected']['out_of_road_rate'] == 0.5
    assert sup['selected']['success_rate'] == 0.25
    assert sup['scenarios_with_lower_intervention_fraction'] == 37
    assert sup['intervention_fraction_relative_change'] < 0


def test_v26_nonpromoted_candidate_not_added_to_curated_registry():
    registry = _load('models/registry.json')
    paths = {str(item.get('path', '')) for item in registry.values()}
    assert not any('v2_6' in path or 'dagger_balanced' in path for path in paths)
